"""Flow Policy Optimization (FPO) for visual RL in ManiSkill.

Implements a flow-based policy with a lightweight conditional flow matching (CFM) objective
and PPO-style clipped surrogate using the FPO ratio.

Notes
- Ratio r_hat = exp( L_CFM(old) - L_CFM(cur) ) estimated with N_mc samples of (tau, eps)
- Action sampling via integrating the learned velocity field from tau=1 to tau=0 (Euler steps)
- Critic/value trained exactly like PPO for GAE advantage estimation

This file mirrors the structure of rl/ppo_rgb.py so scripts/logging are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
import random
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal

# ManiSkill specific imports
import mani_skill.envs  # noqa: F401 - ensure envs are registered
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


# --------- Args ---------
@dataclass
class FPOArgs:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    wandb_group: str = "FPO"
    capture_video: bool = True
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None
    render_mode: str = "all"

    # Environment
    env_id: str = "PickCube-v1"
    env_kwargs: dict = field(default_factory=dict)
    include_state: bool = True

    # Training horizon and rollout
    total_timesteps: int = 10_000_000
    num_envs: int = 512
    num_eval_envs: int = 8
    num_steps: int = 50
    num_eval_steps: int = 50
    partial_reset: bool = True
    eval_partial_reset: bool = False
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = None

    # Optimization
    learning_rate: float = 3e-4
    num_minibatches: int = 32
    update_epochs: int = 4
    clip_coef: float = 0.2  # epsilon_clip
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    reward_scale: float = 1.0
    eval_freq: int = 25
    save_train_video_freq: Optional[int] = None

    # GAE
    gamma: float = 0.9
    gae_lambda: float = 0.9

    # FPO-specific
    n_mc: int = 1           # number of (tau, eps) per transition for ratio
    sampler_steps: int = 8  # Euler steps for flow sampler from tau=1->0

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


# --------- Utilities ---------

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def sinusoidal_embedding(x: torch.Tensor, dims: int = 16) -> torch.Tensor:
    """Sinusoidal positional embedding for scalar tau in [0,1].
    Args:
        x: (..., 1) tensor
        dims: half of the embedding size (sine/cosine pairs)
    Returns: (..., 2*dims)
    """
    device = x.device
    half_dim = dims
    freqs = torch.exp(
        torch.linspace(0, np.log(1000), steps=half_dim, device=device)
    )  # geometric progression
    args = x * freqs  # broadcast
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


def schedule_alpha_sigma(tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple VP-like schedule using alpha_bar = 1 - tau.
    a_tau = sqrt(alpha_bar) * a_0 + sqrt(1-alpha_bar) * eps
    => alpha = sqrt(1-tau), sigma = sqrt(tau)
    tau shape: (..., 1)
    """
    alpha_bar = (1.0 - tau).clamp(0.0, 1.0)
    alpha = torch.sqrt(alpha_bar)
    sigma = torch.sqrt(1.0 - alpha_bar)
    return alpha, sigma


# --------- Model ---------
class NatureCNN(nn.Module):
    def __init__(self, sample_obs: dict):
        super().__init__()
        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]

        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4, 0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        self.extractors = nn.ModuleDict({"rgb": nn.Sequential(cnn, fc)})
        self.out_features += feature_size

        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            self.extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

    def forward(self, observations: dict) -> torch.Tensor:
        encoded = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2) / 255.0
            encoded.append(extractor(obs))
        return torch.cat(encoded, dim=1)


class FlowPolicy(nn.Module):
    """Flow policy that predicts a velocity field v_hat(a_tau, tau, obs) in action space.

    - Provides a sampler integrating from tau=1->0 using Euler steps
    - Provides methods to compute CFM per-sample loss terms for FPO ratio
    - Includes a critic for value function like PPO
    """

    def __init__(self, envs, sample_obs: dict, sampler_steps: int = 8):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        self.sampler_steps = sampler_steps
        action_dim = int(np.prod(envs.unwrapped.single_action_space.shape))

        # Velocity head: takes [features || a_tau || tau_emb] -> v_hat
        tau_emb_dim = 32
        self.tau_emb_dim = tau_emb_dim
        self.velocity_head = nn.Sequential(
            layer_init(nn.Linear(self.feature_net.out_features + action_dim + 2 * tau_emb_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), std=0.01 * np.sqrt(2)),
        )

        # Critic as in PPO
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.feature_net.out_features, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1)),
        )

        # For exploration clipping of actions (respect env bounds)
        self.action_low = torch.as_tensor(envs.unwrapped.single_action_space.low, dtype=torch.float32)
        self.action_high = torch.as_tensor(envs.unwrapped.single_action_space.high, dtype=torch.float32)

    def _velocity(self, features: torch.Tensor, a_tau: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.view(-1, 1)
        tau_emb = sinusoidal_embedding(tau, dims=self.tau_emb_dim)
        x = torch.cat([features, a_tau, tau_emb], dim=-1)
        v = self.velocity_head(x)
        return v

    def sample_action(self, obs: dict) -> torch.Tensor:
        """Deterministic sampler integrating v_hat from tau=1->0 with Euler steps.
        Uses batch-wise integration; returns action in env bounds.
        """
        device = next(self.parameters()).device
        batch = obs["rgb"].shape[0]
        features = self.feature_net(obs)
        action_dim = self.action_low.numel()

        # init from noise at tau=1: alpha=0, sigma=1
        a = torch.randn(batch, action_dim, device=device)
        taus = torch.linspace(1.0, 0.0, steps=self.sampler_steps + 1, device=device)
        for i in range(self.sampler_steps, 0, -1):
            tau_cur = taus[i]
            dt = taus[i] - taus[i - 1]
            v = self._velocity(features, a, torch.full((batch,), tau_cur, device=device))
            a = a + dt * v
        # clamp to action space
        a = torch.max(torch.min(a, self.action_high.to(device)), self.action_low.to(device))
        return a

    def get_value(self, obs: dict) -> torch.Tensor:
        feats = self.feature_net(obs)
        return self.critic(feats)

    # ---- CFM Loss terms for FPO ratio ----
    @torch.no_grad()
    def cfm_loss_per_sample_old(self, obs: dict, action: torch.Tensor, taus: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Compute per-sample CFM loss with current parameters but detached graph (for theta_old snapshot).
        Return shape: (batch, n_mc)
        """
        return self._cfm_loss_per_sample(obs, action, taus, eps, no_grad=True)

    def cfm_loss_per_sample(self, obs: dict, action: torch.Tensor, taus: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self._cfm_loss_per_sample(obs, action, taus, eps, no_grad=False)

    def _cfm_loss_per_sample(self, obs: dict, action: torch.Tensor, taus: torch.Tensor, eps: torch.Tensor, no_grad: bool) -> torch.Tensor:
        """Implements Eq (8): || v_hat(a_tau, tau; o) - (a - eps) ||^2
        - a_tau = alpha(tau) * a + sigma(tau) * eps
        obs: dict with batch dimension B
        action: (B, A)
        taus: (B, n_mc)
        eps: (B, n_mc, A)
        Returns: (B, n_mc)
        """
        B, A = action.shape
        n_mc = taus.shape[1]
        device = action.device
        # Expand
        action_exp = action.unsqueeze(1).expand(B, n_mc, A)
        tau_flat = taus.reshape(-1, 1)
        alpha, sigma = schedule_alpha_sigma(tau_flat)
        alpha = alpha.view(B, n_mc, 1)
        sigma = sigma.view(B, n_mc, 1)
        a_tau = alpha * action_exp + sigma * eps  # (B, n_mc, A)

        # features shared per batch
        features = self.feature_net(obs)  # (B, F)
        features_exp = features.unsqueeze(1).expand(B, n_mc, features.shape[-1]).reshape(B * n_mc, -1)
        a_tau_flat = a_tau.reshape(B * n_mc, A)
        tau_batch = taus.reshape(B * n_mc)

        v_hat = self._velocity(features_exp, a_tau_flat, tau_batch)  # (B*n_mc, A)
        target = (action_exp - eps).reshape(B * n_mc, A)
        loss = torch.sum((v_hat - target) ** 2, dim=-1)  # (B*n_mc)
        return loss.view(B, n_mc)


# --------- Logger ---------
class Logger:
    def __init__(self, log_wandb: bool = False, tensorboard: SummaryWriter | None = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


# --------- Training ---------

def train(args: FPOArgs):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup (match PPO defaults for comparability)
    env_kwargs = dict(
        obs_mode="rgb+segmentation",
        render_mode=args.render_mode,
        sim_backend="physx_cuda",
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    env_kwargs.update(args.env_kwargs)

    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs,
    )
    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )

    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=eval_envs.unwrapped.control_freq,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=eval_envs.unwrapped.control_freq,
            info_on_video=True,
        )

    envs = ManiSkillVectorEnv(
        envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    logger = None
    if not args.evaluate:
        if args.track:
            import wandb

            config = vars(args)
            config["env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            config["eval_env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["fpo", "flow"],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # Storage
    class DictArray(object):
        def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
            self.buffer_shape = buffer_shape
            if data_dict:
                self.data = data_dict
            else:
                assert isinstance(element_space, gym.spaces.dict.Dict)
                self.data = {}
                for k, v in element_space.items():
                    if isinstance(v, gym.spaces.dict.Dict):
                        self.data[k] = DictArray(buffer_shape, v, device=device)
                    else:
                        dtype = (
                            torch.float32
                            if v.dtype in (np.float32, np.float64)
                            else torch.uint8
                            if v.dtype == np.uint8
                            else torch.int16
                            if v.dtype == np.int16
                            else torch.int32
                            if v.dtype == np.int32
                            else v.dtype
                        )
                        self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

        def keys(self):
            return self.data.keys()

        def __getitem__(self, index):
            if isinstance(index, str):
                return self.data[index]
            return {k: v[index] for k, v in self.data.items()}

        def __setitem__(self, index, value):
            if isinstance(index, str):
                self.data[index] = value
            for k, v in value.items():
                self.data[k][index] = v

        @property
        def shape(self):
            return self.buffer_shape

        def reshape(self, shape):
            t = len(self.buffer_shape)
            new_dict = {}
            for k, v in self.data.items():
                if isinstance(v, DictArray):
                    new_dict[k] = v.reshape(shape)
                else:
                    new_dict[k] = v.reshape(shape + v.shape[t:])
            new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
            return DictArray(new_buffer_shape, None, data_dict=new_dict)

    # Initialize policy
    next_obs, _ = envs.reset(seed=args.seed)
    agent = FlowPolicy(envs, sample_obs=next_obs, sampler_steps=args.sampler_steps).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Optionally load checkpoint
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # Rollout buffers
    obs_buf = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    log_ratios = torch.zeros((args.num_steps, args.num_envs), device=device)  # store r_hat for logging only
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    # FPO-specific storage: taus and eps per transition
    taus_buf = torch.zeros((args.num_steps, args.num_envs, args.n_mc), dtype=torch.float32, device=device)
    eps_buf = torch.zeros((args.num_steps, args.num_envs, args.n_mc, envs.single_action_space.shape[0]), dtype=torch.float32, device=device)
    cfm_old_buf = torch.zeros((args.num_steps, args.num_envs, args.n_mc), dtype=torch.float32, device=device)

    global_step = 0
    start_time = time.time()
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        # Evaluation
        if iteration % args.eval_freq == 1:
            agent.eval()
            with torch.no_grad():
                eval_obs, _ = eval_envs.reset()
                successes = []
                for _ in range(args.num_eval_steps):
                    eval_actions = agent.sample_action(eval_obs)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_actions)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        if "success" in eval_infos["final_info"]:
                            successes.append(eval_infos["final_info"]["success"][mask].float().mean())
                if logger is not None and len(successes) > 0:
                    logger.add_scalar("eval/success", torch.stack(successes).mean().item(), global_step)
            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")

        # Rollout
        agent.train()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                act = agent.sample_action(next_obs)
                val = agent.get_value(next_obs).flatten()
            actions[step] = act
            values[step] = val

            # Sample MC pairs and precompute old CFM losses using a frozen copy
            with torch.no_grad():
                taus = torch.rand((args.num_envs, args.n_mc), device=device)
                eps = torch.randn((args.num_envs, args.n_mc, actions.shape[-1]), device=device)
                cfm_old = agent.cfm_loss_per_sample_old(next_obs, act, taus, eps)
                taus_buf[step] = taus
                eps_buf[step] = eps
                cfm_old_buf[step] = cfm_old

            next_obs, reward, terminations, truncations, infos = envs.step(act)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

        # Compute GAE advantages and returns
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * next_not_done - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
                )
            returns = advantages + values

        # Flatten batch
        b_obs = obs_buf.reshape((-1,))
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_taus = taus_buf.reshape((-1, args.n_mc))
        b_eps = eps_buf.reshape((-1, args.n_mc, envs.single_action_space.shape[0]))
        b_cfm_old = cfm_old_buf.reshape((-1, args.n_mc))

        # Policy/value update
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Compute current CFM losses and the FPO ratio
                cfm_cur = agent.cfm_loss_per_sample(b_obs[mb_inds], b_actions[mb_inds], b_taus[mb_inds], b_eps[mb_inds])
                # L_CFM = mean over n_mc
                L_old = torch.mean(b_cfm_old[mb_inds], dim=-1)
                L_cur = torch.mean(cfm_cur, dim=-1)
                ratio = torch.exp(L_old - L_cur)  # shape (mb,)

                mb_adv = b_advantages[mb_inds]
                # Clipped surrogate
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * mb_adv
                policy_loss = -torch.mean(torch.minimum(unclipped, clipped))

                # Value loss
                new_values = agent.get_value(b_obs[mb_inds]).view(-1)
                v_loss = 0.5 * torch.mean((new_values - b_returns[mb_inds]) ** 2)

                loss = policy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        if logger is not None:
            sps = int(global_step / (time.time() - start_time))
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("losses/policy", policy_loss.item(), global_step)
            logger.add_scalar("losses/value", v_loss.item(), global_step)

    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None:
        logger.close()
