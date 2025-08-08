## PPO vs FPO: Intuition and Math from First Principles

### Who this is for
- Practitioners who know PPO and want a principled way to use expressive flow/diffusion policies without computing exact likelihoods.
- Readers who want the core math linking flow matching, ELBOs, and a PPO-style surrogate.

### TL;DR
- **PPO**: Maximizes expected return with a clipped likelihood-ratio surrogate; assumes you can compute πθ(a|o) for sampled actions (e.g., Gaussian policies).
- **FPO**: Keeps the PPO surrogate but replaces the intractable likelihood ratio with an **ELBO-driven ratio** derived from **conditional flow matching (CFM)**. Intuitively, it steers the policy’s probability flow toward high-advantage actions.
- **Why FPO can help**: Flow/diffusion policies can be far more expressive (multimodal, non-Gaussian). FPO preserves PPO’s stability while enabling these richer policies and fast deterministic sampling.

---

## Background: Policy gradients and PPO
We want to maximize expected return J(θ) with a policy πθ. A classic on-policy objective uses the (surrogate) policy gradient with advantages $\hat A_t$:

$$\max_\theta\; \mathbb{E}_{a_t \sim \pi_\theta(\cdot|o_t)}\; [\, \log \pi_\theta(a_t|o_t)\, \hat A_t\,].$$

Large steps destabilize training. **PPO** introduces a trust region via the clipped ratio $r(\theta)$:

$$r(\theta) = \frac{\pi_\theta(a_t|o_t)}{\pi_{\text{old}}(a_t|o_t)}.$$

Its clipped surrogate is

$$\max_\theta\; \mathbb{E}_{a_t \sim \pi_{\text{old}}}\; \big[\, \min\big( r(\theta)\, \hat A_t,\; \mathrm{clip}(r(\theta), 1-\varepsilon, 1+\varepsilon)\, \hat A_t \big)\,\big].$$

This requires computing (or at least evaluating) $\pi_\theta(a|o)$, which is easy for Gaussians but **hard for flow/diffusion models**.

---

## Flow/diffusion policies and conditional flow matching
Flow/diffusion policies generate an action by transforming noise through a time-indexed process (denoising or continuous flow) conditioned on the observation $o$. We index that process with a flow timestep $\tau \in [0, 1]$ (not the environment time).

A convenient noise-conditioning for actions uses a schedule $\alpha_\tau, \sigma_\tau$ to define a noisy action:

$$ a^\tau_t \;=\; \alpha_\tau\, a_t \; + \; \sigma_\tau\, \varepsilon,\qquad \varepsilon \sim \mathcal N(0, I). $$

A flow policy learns a velocity field $\hat v_\theta(a^\tau_t, \tau; o_t)$ that “points back” toward the clean action. The **conditional flow matching (CFM)** per-sample loss is a simple denoising MSE:

$$ \ell_\theta(\tau, \varepsilon) \;=\; \big\|\, \hat v_\theta(a^\tau_t,\tau; o_t)\; -\; (a_t - \varepsilon)\,\big\|_2^2. $$

A Monte Carlo estimate of the CFM loss for a fixed $(o_t, a_t)$ averages over $N_{\text{mc}}$ pairs $(\tau_i, \varepsilon_i)$:

$$ \hat L_{\text{CFM},\theta}(a_t; o_t) \;=\; \frac{1}{N_{\text{mc}}} \sum_{i=1}^{N_{\text{mc}}} \ell_\theta(\tau_i, \varepsilon_i). $$

This training objective is standard in flow/diffusion literature and closely connected to ELBO maximization.

---

## From ELBO to a surrogate likelihood ratio
Computing exact $\log \pi_\theta(a|o)$ for flow/diffusion models is often **intractable**. Instead, these models frequently optimize an **ELBO** as a proxy for log-likelihood. For a broad class of schedules (including the diffusion schedule), there is a key identity:

$$ L^w_\theta(a_t) \;=\; -\; \text{ELBO}_\theta(a_t) \; + \; c, $$

where $L^w_\theta$ is a weighted denoising loss (CFM is a special case), and $c$ is constant w.r.t. $\theta$. This means reducing the denoising loss increases the ELBO and (loosely) the modeled likelihood of $a_t$.

Define the **FPO ratio** as the ratio of ELBOs under current and old policies:

$$ r_{\text{FPO}}(\theta) \;=\; \exp\big( \text{ELBO}_\theta(a_t|o_t) - \text{ELBO}_{\text{old}}(a_t|o_t) \big). $$

Using the link above, we can estimate this ratio with the same Monte Carlo terms used for CFM:

$$ \hat r_{\text{FPO}}(\theta) \;=\; \exp\big( \hat L_{\text{CFM},\text{old}}(a_t; o_t) - \hat L_{\text{CFM},\theta}(a_t; o_t) \big). $$

One can further decompose the true quantity as

$$ r_{\text{FPO}}(\theta) \;=\; \underbrace{\frac{\pi_\theta(a_t|o_t)}{\pi_{\text{old}}(a_t|o_t)}}_{\text{likelihood ratio}}\; \times\; \underbrace{\frac{\exp(D_{\mathrm{KL}}(\text{old}))}{\exp(D_{\mathrm{KL}}(\theta))}}_{\text{inv. KL gap correction}}, $$

so maximizing $r_{\text{FPO}}$ both increases modeled likelihood and reduces the ELBO gap—both beneficial for policy optimization.

Practical estimator (stored MC): we sample $(\tau, \varepsilon)$ once during rollout and reuse the same draws for both $\theta_{\text{old}}$ and $\theta$ inside each minibatch update, which reduces variance and respects the PPO-style trust region.

---

## FPO objective: PPO’s surrogate, new ratio
FPO simply plugs the surrogate ratio into PPO’s clipped objective:

$$ \max_\theta\; \mathbb{E}\; \big[\, \min\big( \hat r_{\text{FPO}}(\theta)\, \hat A_t,\; \mathrm{clip}(\hat r_{\text{FPO}}(\theta), 1-\varepsilon, 1+\varepsilon)\, \hat A_t \big)\,\big]. $$

Everything else—GAE, value learning, batching, clipping—follows PPO.

- With $N_{\text{mc}} = 1$ the ratio is biased high by Jensen’s inequality, but the gradient direction remains **unbiased**: $\mathbb{E}[\,-\nabla_\theta \ell_\theta\,] = \nabla_\theta \text{ELBO}_\theta$. Empirically, even small $N_{\text{mc}}$ works well.
- You can increase $N_{\text{mc}}$ for a better ratio estimate at extra compute cost.

---

## Intuition: “Push probability flow toward good actions”
- In PPO with Gaussians, increasing $\log \pi_\theta(a|o)$ for positive-advantage $a$ literally makes that action more likely under a simple parametric family.
- In FPO, we do not evaluate likelihoods directly. Instead, we reduce a denoising loss $L_{\text{CFM}}$ that is equivalent (up to constants) to increasing the ELBO. This **redirects the learned flow** so that starting from noise, the sampler moves toward actions like $a$ more decisively.
- The clipped surrogate still guards against destructive updates, giving PPO-like stability with a richer policy class.

---

## Why FPO can be better than Gaussian PPO
- **Expressiveness**: Flow/diffusion policies capture complex, multimodal action distributions that Gaussians cannot. This matters when “good” actions form multiple modes or live on curved manifolds.
- **Sampling flexibility**: Deterministic samplers (and higher-order integrators) can generate actions in a few steps; no requirement to expand the RL horizon as in denoising-MDP approaches.
- **Stable updates**: The clipped surrogate reuses PPO’s trust-region intuition, while the ratio comes from ELBO/CFM, avoiding intractable likelihoods.
- **Empirical wins**: In tasks with multi-goal or multi-strategy solutions, FPO can outperform Gaussian PPO with similar compute.

### When PPO may be preferable
- **Simple action landscapes**: If a single-mode Gaussian is sufficient, PPO’s simplicity and speed may be ideal.
- **Tight compute budgets**: FPO adds per-sample CFM terms (sampling $(\tau, \varepsilon)$, evaluating $\hat v_\theta$). Though lightweight, it’s extra work.
- **Maturity**: PPO has battle-tested implementations; FPO is newer and may need tuning (sampler steps, $N_{\text{mc}}$).

---

## Practical guidance
- **Keep everything else identical** (envs, GAE, evaluation) to make a clean comparison.
- **Key knobs**
  - $N_{\text{mc}}$: start at 1; try 2–4 if noisy.
  - Sampler steps (τ from 1→0): start 8; try 4–16. More steps can stabilize at extra latency.
  - Clip coefficient $\varepsilon$: similar to PPO defaults (e.g., 0.2).
  - Learning rate: start from PPO LR; reduce slightly if unstable.
- **Logging**: track eval success/return, policy/value losses, SPS. Compare wall-clock vs performance across PPO and FPO.

---

## Algorithm sketch
1. Collect on-policy rollouts with the flow policy (any sampler; often deterministic Euler steps). Compute advantages (e.g., GAE).
2. For each $(o_t, a_t)$, sample and store $N_{\text{mc}}$ pairs $(\tau_i, \varepsilon_i)$ and compute $\ell_{\text{old}}(\tau_i, \varepsilon_i)$.
3. In each update minibatch, recompute $\ell_\theta(\tau_i, \varepsilon_i)$, form
   $\hat r_{\text{FPO}} = \exp\big(\overline{\ell}_{\text{old}} - \overline{\ell}_\theta\big)$, and apply the PPO clipped surrogate.
4. Train the critic with standard value regression; repeat.

---

## References
- PPO: Schulman et al., “Proximal Policy Optimization Algorithms” (2017). [arXiv link](https://arxiv.org/abs/1707.06347)
- Flow Policy Optimization (project page): [flowreinforce.github.io](https://flowreinforce.github.io/)
- Conditional flow matching and ELBO links:
  - Kingma & Gao, “Understanding diffusion objectives...” (2023). [arXiv link](https://arxiv.org/abs/2302.00783)
  - Lipman et al., “Flow Matching for Generative Modeling” (2023). [arXiv link](https://arxiv.org/abs/2210.02747)
