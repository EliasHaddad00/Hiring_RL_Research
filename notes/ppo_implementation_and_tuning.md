# Implementing and tuning masked PPO (MaskablePPO) on a custom Gymnasium env with a variable set of candidates

Research date: 2026-11-12. Scope: implementation, debugging, tuning and logging of the PPO agent for the "hiring as a race against poaching" simulator (Discrete(2N+1) actions, N candidate slots, CPU only, 5-week course project).

Conventions used below:
- "Local test" means I installed the current releases in a sandbox (Python 3.13.18, 4 cores, gymnasium 1.4.0, stable_baselines3 2.9.2, sb3_contrib 2.9.2, torch 2.15.0, numpy 2.5.4) and ran a toy stand-in for the hiring simulator (10 slots, 5 features per candidate, 3 global features, 21 actions). These are first-hand observations, not web sources, and the toy env is not the team's simulator.
- Where a web fetch returned only an abstract or a secondary source, this is said in the bullet and repeated under Gaps.

---

## 1. Gymnasium custom environment: required API, checker, vectorized envs, wrappers, current versions

### Takeaway
As of November 2026 the current stack is Gymnasium 1.4.0, Stable-Baselines3 2.9.2 and SB3-Contrib 2.9.2, and they install and run together on Python 3.13 (verified locally). Write a plain single-env `gym.Env` with an `action_masks()` method, end the episode at day 60 with `terminated=True` (the horizon is part of the task and "days left" is observed), and start with `DummyVecEnv`, which was 2-3x faster than `SubprocVecEnv` on a cheap numpy env in my test.

### Cited Findings

Versions and compatibility
- Gymnasium v1.4.0 was released 2026-10-05; v1.3.0 on 2026-04-22; v1.2.3 on 2025-12-18 — [Gymnasium release notes](https://gymnasium.farama.org/gymnasium_release_notes/)
- Gymnasium 1.4.0 added a check that "`reset` and `step` return distinct observation and info objects each timestep" to the environment checker, and fixed vector `RecordEpisodeStatistics` / `NormalizeReward` under `SAME_STEP` autoreset — [Gymnasium release notes](https://gymnasium.farama.org/gymnasium_release_notes/)
- Stable-Baselines3 2.9.2 was released 2026-10-26; 2.9.0 on 2026-06-15; 2.8.0 on 2026-04-01 — [SB3 changelog](https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html)
- SB3 2.9.0 relaxed the Gymnasium requirement "from `gymnasium>=0.29.1,<1.3.0` to `gymnasium>=0.29.1,<2.0`", raised the minimum `torch` from 2.3 to 2.8, and made `pandas` and `matplotlib` optional (pandas is needed only for loading results and plotting) — [SB3 changelog](https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html)
- SB3 2.8.0 dropped Python 3.9 and added official Python 3.13 support; "v2.9.2 will be the last one supporting Python 3.10", and the maintainers recommend Python >= 3.11 — [SB3 changelog](https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html)
- SB3 2.8.0 fixed "`MaskablePPO` and `RecurrentPPO` inaccurate `n_updates` counting when `target_kl` early exits the training loop" — [SB3 changelog](https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html)
- The SB3-Contrib docs and the RL Baselines3 Zoo docs both show version 2.9.2 — [SB3-Contrib MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html), [RL Zoo tuning docs](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html)
- Local test: `pip install sb3-contrib` on 2026-11-12 resolved to gymnasium 1.4.0, stable_baselines3 2.9.2, sb3_contrib 2.9.2, torch 2.15.0, numpy 2.5.4 on Python 3.13.18, and MaskablePPO trained, evaluated, saved and loaded without errors.

Required env API
- The env subclasses `gym.Env` and defines `self.observation_space` and `self.action_space`; `reset(self, seed=None, options=None)` must call `super().reset(seed=seed)` first (it seeds `self.np_random`) and return `(observation, info)`; `step(self, action)` returns `(observation, reward, terminated, truncated, info)` — [Gymnasium: create a custom env](https://gymnasium.farama.org/introduction/create_custom_env/)
- Registration is optional: `gym.register(id=..., entry_point=..., max_episode_steps=300)`; `max_episode_steps` prevents infinite episodes — [Gymnasium: create a custom env](https://gymnasium.farama.org/introduction/create_custom_env/)
- Gymnasium's checker is `from gymnasium.utils.env_checker import check_env`, described as catching "many common issues" — [Gymnasium: create a custom env](https://gymnasium.farama.org/introduction/create_custom_env/)
- SB3's advice for custom envs: "always normalize your observation space if you can, i.e. if you know the boundaries"; start with a shaped reward and a simplified problem; debug with random actions; run `check_env`; keep the Markov property; return `truncated = True` for timeouts — [SB3 RL tips and tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- For fixed-horizon tasks, "the terminations due to time limits are in fact part of the environment" and the agent should receive the remaining time as input; bootstrapping at the limit is for time-unlimited tasks only — [Pardo et al., Time Limits in RL, arXiv 1712.00378](https://arxiv.org/abs/1712.00378) (abstract only read)
- The MaskablePPO docs warn that `check_env` calls `step` with random actions sampled from the action space and ignores the mask — [SB3-Contrib MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- Local test: SB3's `check_env` did raise my env's own `assert mask[action]` ("invalid action 15"). The env must tolerate invalid actions during the check (e.g. treat them as a no-op behind a `strict=False` flag) or the check must be run on a variant without the assert.

SB3 vectorized envs
- SB3 `VecEnv.step` returns a 4-tuple `obs, rewards, dones, infos` with `dones = terminated or truncated`; `reset()` returns only `obs`; envs are reset automatically at episode end, the true last observation is in `infos[i]["terminal_observation"]`, and `infos[i]["TimeLimit.truncated"] = truncated and not terminated` — [SB3 vectorized envs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- `DummyVecEnv` is "useful for computationally simple environment such as `Cartpole-v1`, as the overhead of multiprocess or multithread outweighs the environment computation time"; `SubprocVecEnv` gives "significant speed up when the environment is computationally complex"; the number of envs "should not exceed the number of logical cores" — [SB3 vectorized envs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- With `SubprocVecEnv`, code must be wrapped in `if __name__ == "__main__":` when using `forkserver` or `spawn` (the default on Windows); `fork` "is not thread safe and can create deadlocks" — [SB3 vectorized envs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- The PPO page says "PPO is meant to be run primarily on the CPU, especially when you are not using a CNN" and recommends `SubprocVecEnv` over the default `DummyVecEnv` — [SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html). This general recommendation conflicts with the vec-env guide's statement above for cheap envs, and with my measurement below.
- Local test (4 cores, toy env, MaskablePPO `MlpPolicy`, 20,480 steps, `n_epochs=4`, `batch_size=256`): `DummyVecEnv` x8 = about 7,000 steps/s; `SubprocVecEnv` x4 = about 2,000 steps/s; `SubprocVecEnv` x8 = about 3,000 steps/s.

Normalization and monitoring wrappers
- `VecNormalize` defaults: `training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-08`; statistics are saved with `save(path)` and restored with `VecNormalize.load(path, venv)` — [SB3 vectorized envs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- `VecMonitor(venv, filename=None, info_keywords=())` records episode reward, length and time — [SB3 vectorized envs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- `rollout/ep_rew_mean` requires a `Monitor` wrapper, which `make_vec_env` adds automatically — [SB3 logger docs](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)
- SB3 guidance: "always normalize the input to the agent (e.g. using `VecNormalize` for PPO/A2C)" — [SB3 RL tips and tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

### Inferences
- Day 60 should be `terminated=True`, not `truncated=True`: the window is part of the problem and "days left" is in the observation, which is exactly Pardo et al.'s fixed-horizon case. Using `truncated=True` would make SB3 bootstrap from a value of a state that has no future.
- Because Gymnasium 1.4.0's checker now requires distinct observation objects per step, return a fresh array (`obs.copy()` or build a new one) rather than a view of internal simulator state.
- `VecNormalize(norm_obs=True)` keeps separate running statistics per observation dimension, so slot 3's "posterior mean" would be scaled differently from slot 7's. That breaks exact weight sharing across slots for the index-structured policy. Prefer fixed, hand-chosen scaling inside the env (as SB3 suggests when bounds are known), identical for every slot, and use `VecNormalize` only for rewards (`norm_obs=False, norm_reward=True`) if needed.
- When `VecNormalize(norm_reward=True)` is used, set its `gamma` to the same value as PPO's `gamma` (its default is 0.99), and at evaluation load the saved statistics with `training=False` and `norm_reward=False` so reported returns are in raw units. The attribute names are documented; the evaluation procedure itself is not spelled out on the page I read.
- The team's simulator is already batched in numpy. The largest remaining speed-up would be a custom SB3 `VecEnv` subclass that steps all envs in one numpy call instead of looping over single envs. It must implement `env_method("action_masks")` returning one mask per env, because MaskablePPO collects masks with `np.stack(env.env_method("action_masks"))` (seen in the `sb3_contrib/common/maskable/utils.py` traceback in my local test). I did not build or time this.

### Gaps
- I did not read the Gymnasium `Env` API reference page itself, so the formal definitions of `terminated` vs `truncated` come from the SB3 docs and Pardo et al. rather than from Gymnasium's text.
- I did not find the list of what Gymnasium's `check_env` validates beyond the 1.4.0 release-note item.
- Supported Python/NumPy ranges for Gymnasium 1.4.0 are not stated on the release-notes page; only the working combination above was verified by installation.
- The custom batched `VecEnv` route is untested; its speed-up is unknown.

---

## 2. MaskablePPO in SB3-Contrib: masks, evaluation, limitations, and the index-structured custom policy

### Takeaway
MaskablePPO takes masks from an `action_masks()` method on the env (or an `ActionMasker` wrapper for non-subprocess envs) and needs its own `evaluate_policy` and `MaskableEvalCallback`. The index-structured policy is straightforward in SB3: subclass `MaskableActorCriticPolicy`, override `_build_mlp_extractor` with a module that emits the 2N+1 logits directly, and replace `action_net` with `nn.Identity()`; I ran this and confirmed training, permutation-equivariance and save/load.

### Cited Findings

Mask supply and evaluation
- MaskablePPO "adds invalid action masking to PPO" and otherwise behaves like core PPO; masks are boolean with `True` = valid — [SB3-Contrib MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- If the env has an `action_masks` method it is used automatically; otherwise wrap with `ActionMasker(env, mask_fn)` where `mask_fn(env)` returns the mask — [SB3-Contrib MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- "In order to use SubprocVecEnv with MaskablePPO, you must implement the action_masks inside the environment"; `ActionMasker` cannot be used in that case — [SB3-Contrib MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- Inference-time pattern: `action_masks = get_action_masks(env)` then `model.predict(obs, action_masks=action_masks)`; masking is optional at test time and silently omitted if not passed — [SB3-Contrib MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- Use `evaluate_policy` from `sb3_contrib.common.maskable.evaluation` and `MaskableEvalCallback` from `sb3_contrib.common.maskable.callbacks`, not the base SB3 versions — [SB3-Contrib MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- Limitations table: recurrent policies No; multiprocessing Yes; action spaces Discrete, MultiDiscrete, MultiBinary supported, Box and Dict actions not; all observation spaces supported — [SB3-Contrib MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- Constructor defaults: `learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_kl=None, stats_window_size=100, tensorboard_log=None, policy_kwargs=None, verbose=0, seed=None, device='auto'`; policy aliases are `MlpPolicy` = `MaskableActorCriticPolicy`, plus `CnnPolicy` and `MultiInputPolicy` — [SB3-Contrib MaskablePPO docs](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)

How masking is implemented
- Masked logits are replaced with `-1e8` (not `-inf`) via `th.where(masks, original_logits, HUGE_NEG)`; the entropy method zeroes the masked entries of `p * log p` so invalid actions contribute nothing — [sb3_contrib distributions.py source](https://raw.githubusercontent.com/Stable-Baselines-Team/stable-baselines3-contrib/master/sb3_contrib/common/maskable/distributions.py)
- For `Discrete` the mask must reshape to `(batch, action_dim)`. For `MultiDiscrete` the flat mask has width `sum(nvec)` and is split per sub-action, i.e. each dimension is masked independently — [sb3_contrib distributions.py source](https://raw.githubusercontent.com/Stable-Baselines-Team/stable-baselines3-contrib/master/sb3_contrib/common/maskable/distributions.py)
- In the policy, masks are applied to the distribution object after it is built: `distribution = self._get_action_dist_from_latent(latent_pi); if action_masks is not None: distribution.apply_masking(action_masks)`, in `forward`, `evaluate_actions` and `get_distribution` — [sb3_contrib maskable policies source (v2.7.0)](https://sb3-contrib.readthedocs.io/en/v2.7.0/_modules/sb3_contrib/common/maskable/policies.html)

Custom-policy hooks
- `MaskableActorCriticPolicy.__init__` signature: `(observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh, ortho_init=True, features_extractor_class=FlattenExtractor, features_extractor_kwargs=None, share_features_extractor=True, normalize_images=True, optimizer_class=th.optim.Adam, optimizer_kwargs=None)` — [sb3_contrib maskable policies source (v2.7.0)](https://sb3-contrib.readthedocs.io/en/v2.7.0/_modules/sb3_contrib/common/maskable/policies.html)
- `action_net` is created in `_build()` as `self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor.latent_dim_pi)` (an `nn.Linear(latent_dim, action_dim)`) and applied in `_get_action_dist_from_latent`; every action path goes through that method — [sb3_contrib maskable policies source (v2.7.0)](https://sb3-contrib.readthedocs.io/en/v2.7.0/_modules/sb3_contrib/common/maskable/policies.html)
- With `ortho_init=True`, gains are sqrt(2) for the features extractor and `mlp_extractor`, 0.01 for `action_net`, 1 for `value_net` — [sb3_contrib maskable policies source (v2.7.0)](https://sb3-contrib.readthedocs.io/en/v2.7.0/_modules/sb3_contrib/common/maskable/policies.html)
- Official SB3 recipe for a custom actor/critic body: write an `nn.Module` exposing `latent_dim_pi`, `latent_dim_vf`, `forward`, `forward_actor`, `forward_critic`; subclass the policy, override `_build_mlp_extractor` to instantiate it, and pass the policy class to the algorithm. The docs' example disables `ortho_init` — [SB3 custom policy guide](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
- A custom features extractor subclasses `BaseFeaturesExtractor(observation_space, features_dim)` and is passed with `policy_kwargs=dict(features_extractor_class=..., features_extractor_kwargs=dict(...))` — [SB3 custom policy guide](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
- `net_arch=dict(pi=[...], vf=[...])` sets separate actor/critic sizes; a list `[128, 128]` means the same sizes for both; the default for 1-D observations is two layers of 64 units; "an extra linear layer will be added on top" for the outputs — [SB3 custom policy guide](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)

Local test results for the index-structured policy (code below)
- Local test: the policy below trained under `MaskablePPO` with `make_vec_env(ToyHiringEnv, n_envs=8)` (masks were found through the `Monitor` wrapper; `get_action_masks(venv)` returned shape `(8, 21)`).
- Local test: permuting the candidate slots of an observation permuted the per-candidate logits identically and left the end-day logit unchanged (`np.allclose`, tolerance 1e-5).
- Local test: `model.save` / `MaskablePPO.load` round-tripped and produced identical deterministic actions, provided `_get_constructor_parameters` is overridden as shown.
- Local test: the index policy had 12,868 parameters and ran at about 3,900 steps/s (8 envs, `n_steps=256`, `batch_size=256`, `n_epochs=4`); on the toy env `rollout/ep_rew_mean` rose from -0.25 to 2.19 within 40,000 steps.
- Local test: an all-`False` mask raised no error; with every logit at `-1e8` the distribution becomes uniform over all actions, including invalid ones.
- Local test: `get_action_masks(venv)` fails with `AttributeError` if called before the first `venv.reset()` when the mask depends on state created in `reset`.

Tested code (observation = N slots x F features flattened, then G global features; action `a < 2N`: candidate `a // 2`, type `a % 2` with 0 = interview, 1 = offer; action `2N` = end day):

```python
import numpy as np, torch as th
from torch import nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

def mlp(sizes, act=nn.Tanh):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), act()]
    return nn.Sequential(*layers[:-1])            # no activation after the last layer

class IndexExtractor(nn.Module):
    """Actor: one shared net per candidate -> 2 logits; end-day logit from globals only.
       Critic: unrestricted MLP on the full observation."""
    def __init__(self, features_dim, n_slots, f_cand, f_glob, hidden=64, vf_hidden=64):
        super().__init__()
        assert features_dim == n_slots * f_cand + f_glob
        self.n, self.fc, self.fg = n_slots, f_cand, f_glob
        self.latent_dim_pi = 2 * n_slots + 1      # the "latent" IS the logit vector
        self.latent_dim_vf = vf_hidden
        self.cand_net = mlp([f_cand + f_glob, hidden, hidden, 2])
        self.end_net = mlp([f_glob, hidden, 1])
        self.vf_net = nn.Sequential(mlp([features_dim, vf_hidden, vf_hidden]), nn.Tanh())
        for m in self.modules():                  # PPO init: orthogonal, gain sqrt(2)
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2)); nn.init.zeros_(m.bias)
        for last in (self.cand_net[-1], self.end_net[-1]):
            nn.init.orthogonal_(last.weight, gain=0.01)   # small gain on policy outputs

    def forward_actor(self, features):
        b = features.shape[0]
        cand = features[:, : self.n * self.fc].reshape(b, self.n, self.fc)
        glob = features[:, self.n * self.fc:]
        x = th.cat([cand, glob.unsqueeze(1).expand(-1, self.n, -1)], dim=-1)
        cand_logits = self.cand_net(x).reshape(b, 2 * self.n)   # [int_0, off_0, int_1, off_1, ...]
        return th.cat([cand_logits, self.end_net(glob)], dim=1)

    def forward_critic(self, features):
        return self.vf_net(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

class IndexPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, n_slots=10, f_cand=5, f_glob=3, hidden=64, **kwargs):
        self._idx_cfg = dict(n_slots=n_slots, f_cand=f_cand, f_glob=f_glob, hidden=hidden)
        kwargs["ortho_init"] = False              # SB3 would put gain sqrt(2) on our logit layer
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = IndexExtractor(self.features_dim, **self._idx_cfg)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = nn.Identity()           # drop SB3's Linear(2N+1, 2N+1), which would mix slots
        nn.init.orthogonal_(self.value_net.weight, gain=1.0); nn.init.zeros_(self.value_net.bias)
        self.optimizer = self.optimizer_class(    # rebuild so the discarded layer is not optimised
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):        # needed for save/load
        data = super()._get_constructor_parameters()
        data.update(self._idx_cfg)
        return data

model = MaskablePPO(IndexPolicy, venv, policy_kwargs=dict(n_slots=10, f_cand=5, f_glob=3),
                    n_steps=256, batch_size=256, n_epochs=4, gamma=0.999, ent_coef=0.01, device="cpu")
```

### Inferences
- The one thing that silently destroys the index structure is leaving SB3's default `action_net` in place: it is a dense `Linear(latent, 2N+1)` and mixes all candidates. Replacing it with `nn.Identity()` (as above) or overriding `_get_action_dist_from_latent` are the two clean fixes.
- The unrestricted variant (b) needs no custom code: `MaskablePPO("MlpPolicy", env, policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))`. Attention/Deep Sets variants reuse the same `_build_mlp_extractor` hook.
- For a clean ablation, keep the critic identical across variants (here an unrestricted MLP) so only the actor's structure differs; the critic is not used at deployment, so giving it full information does not violate the index restriction. State this choice in the report.
- The end-day logit is computed from global features only in my implementation. This is a real modelling restriction (the decision to stop for the day cannot depend on who is in the pool except through the softmax competition with candidate logits) and should be stated and possibly ablated.
- Because an all-`False` mask fails silently, keep "end day" always valid and add an assertion `mask.any()` in the env.
- Always pass masks in evaluation code. Forgetting `action_masks=` in `predict` produces invalid actions without an error from SB3.
- Add a permutation-equivariance unit test (as in my local test) to the repo; it is the cheapest proof that variant (a) is what it claims to be.

### Gaps
- I searched for GitHub issues on MaskablePPO custom policies with per-action shared networks and found none that address this design directly; the guidance above rests on the official custom-policy guide, the source code and my own test.
- The policy source I read is the v2.7.0 docs page; I did not diff it against 2.9.2, although the tested code runs on 2.9.2.
- CleanRL alternative: the CleanRL PPO docs page lists `ppo.py` and variants but no masked-PPO script, and does not state maintenance status ([CleanRL PPO docs](https://docs.cleanrl.dev/rl-algorithms/ppo/)). Masked single-file PPO exists in the authors' companion repos ([vwxyzjn/ppo-implementation-details](https://www.Github.com/vwxyzjn/ppo-implementation-details), and vwxyzjn/invalid-action-masking named on the [arXiv page of Huang & Ontañón](https://arxiv.org/abs/2006.14171)), but I did not open those files, so file names and line counts are unverified. My estimate, not a sourced fact: porting means adding a masked categorical class, storing masks in the rollout buffer and passing them to the agent at sampling and update time, on the order of tens of lines; the custom network is easier in CleanRL (plain PyTorch) but you lose SB3's evaluation callbacks, save/load and logger. Given that the SB3 route worked on the first attempt, CleanRL is a fallback rather than a need.
- One sourced data point for the CleanRL route: the entity-based PPO trainer `enn-trainer` used in the cyber-defence paper was "adapted from CleanRL" ([Symes Thompson et al. 2024](https://arxiv.org/pdf/2410.17647)).

---

## 3. Evidence on invalid action masking and on sequential / auto-regressive action decomposition

### Takeaway
Masking is theoretically a valid policy gradient and empirically scales far better than penalising invalid actions; the main risks are sampling with a mask but updating without it, and masks that do not reflect dependencies between sub-actions. A flat Discrete(2N+1) head with one joint mask avoids the dependency problem entirely, so no auto-regressive decomposition is needed at N = 10-20.

### Cited Findings
- Huang & Ontañón prove the masked gradient is a valid policy gradient for the masked policy; masking "can be seen as applying a state-dependent differentiable function during the calculation of the action probability distribution" and "makes the gradient corresponding to the logits of the invalid action to zero" — [Huang & Ontañón 2020, arXiv 2006.14171v3](https://arxiv.org/pdf/2006.14171v3)
- Setup: PPO in μRTS on 4x4, 10x10, 16x16 and 24x24 maps for 500,000 steps; the unit-selection range grows from 16 to 576 — [Huang & Ontañón 2020](https://arxiv.org/pdf/2006.14171v3)
- "Invalid action masking scales well as the number of invalid actions increases": time to first reward stayed at 0.05-0.08% of training across map sizes, while "invalid action penalty does not scale", and a penalty of -1 "seems to have an adverse effect of discouraging exploration" — [Huang & Ontañón 2020](https://arxiv.org/pdf/2006.14171v3)
- "Naive" masking (sample from the masked distribution, update with the unmasked one) showed an average KL divergence "significantly higher than that of any other experiments" and volatile solve times (49.14% of training at 24x24 versus 18.38% for proper masking) — [Huang & Ontañón 2020](https://arxiv.org/pdf/2006.14171v3)
- A policy trained with masking and evaluated without it "is still able to perform well to a certain degree" on small maps but degrades on large ones (return 17.37 at 24x24 with about 150 invalid selections) — [Huang & Ontañón 2020](https://arxiv.org/pdf/2006.14171v3)
- The 37-details post lists invalid action masking as an auxiliary technique: invalid logits are replaced "with negative infinity before passing the logits to softmax", and it was critical for beating past μRTS bots — [Huang et al. 2022, ICLR blog track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)
- PPO's MultiDiscrete handling treats components as independent: prob(a) is the product of per-component probabilities — [Huang et al. 2022, ICLR blog track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)
- Conditional Action Trees paper: masks built per sub-action "do not take into account that the masking of some subactions can depend on the sampled values of others", so agents can still pick unavailable actions; a "collapsed" union-of-masks variant performed similarly to no masking, and full conditional masking was important for efficient training — [Bamford & Ovalle 2021, arXiv 2104.07294](https://arxiv.org/pdf/2104.07294)
- The same paper notes that a flat policy must score every parameter combination (6,382 logits in a BotBowl example), which is the motivation for decomposition in large spaces — [Bamford & Ovalle 2021](https://arxiv.org/pdf/2104.07294)
- An entity-based PPO agent used a two-stage action: a pointer-network-like "select entity" head (query-key dot product with softmax over eligible entities) followed by a global categorical action-type head — [Symes Thompson et al. 2024, arXiv 2410.17647](https://arxiv.org/pdf/2410.17647)
- Kanervisto et al. report that removing domain-specific actions and discretizing continuous actions "can be crucial for successful learning", and that action-space design is "mostly done based on intuition" — [Kanervisto et al. 2020, arXiv 2004.00980](https://arxiv.org/abs/2004.00980) (abstract only read)

### Inferences
- The team's design is the favourable case: 21-41 logits with one joint mask per state. SB3's MultiDiscrete masking is per-dimension and cannot express "interview is invalid for candidate 4 only", so a `MultiDiscrete([N, 2])` formulation would recreate the collapsed-mask failure above. Keep Discrete(2N+1).
- "One candidate, one action per step, then end day" already is a sequential decomposition of the daily combinatorial decision (which subset to interview and whom to offer). Its cost is longer episodes (more steps per day, hence the 100-300 step range) and a longer credit-assignment path, which interacts with the choice of gamma (Section 4). The order of within-day actions is also learned, which adds some redundancy.
- What typically goes wrong with masking in practice, each checkable: mask computed from a stale state (off by one step after auto-reset); mask order not matching the action encoding; masks omitted in evaluation; an all-invalid mask; entropy and KL not comparable across states because the number of valid actions changes.
- A useful ablation for the report, supported by Huang & Ontañón's comparison: masking versus a small invalid-action penalty on a small N.

### Gaps
- I did not read AlphaStar or other primary sources on auto-regressive heads; the auto-regressive evidence here comes from Bamford & Ovalle and the entity-based cyber-defence paper only.
- The venue of Huang & Ontañón is not stated on arXiv; the DOI (10.32473/flairs.v35i.130584) points to FLAIRS-35 (2022), which is an inference from the DOI.
- Within the paper, one number is inconsistent between text and table (penalty time-to-first-reward at 10x10: 3.43% in text versus 1.57% in Table 2 for r = -0.01), so quote the qualitative conclusion rather than that figure.
- No study was found that measures masking on a task with stochastic arrivals/departures of entities like this one.

---

## 4. PPO implementation details and tuning; hyperparameter ranges; principled search

### Takeaway
SB3 already implements the 13 core PPO details, so tuning effort should go to gamma, rollout size, learning rate and entropy coefficient rather than to re-implementing the algorithm. RL Zoo's built-in Optuna search does not list `ppo_mask`, so the practical route is a short custom Optuna script that reuses the Zoo's PPO search space.

### Cited Findings

The 13 core details and their values
- Vectorized architecture (N envs x M steps per update); orthogonal init with gain sqrt(2) for hidden layers, 0.01 for the policy output, 1 for the value output, biases 0; Adam epsilon 1e-5; linear learning-rate annealing to 0 (from 2.5e-4 Atari, 3e-4 MuJoCo); GAE with lambda 0.95 and gamma 0.99; shuffled minibatches (4 minibatches x 4 epochs Atari, 32 x 10 MuJoCo); per-minibatch advantage normalization; clip range 0.1 (Atari) or 0.2 (MuJoCo); value-loss clipping; entropy coefficient 0.01 (Atari) or 0.0 (MuJoCo); global gradient-norm clip 0.5; debug variables; shared 2x64 tanh MLP by default with separate networks as an option — [Huang et al. 2022, The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)
- On value-loss clipping: Engstrom et al. "find no evidence that the value function loss clipping helps with the performance", and Andrychowicz et al. "suggest value function loss clipping even hurts performance" — [Huang et al. 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)
- `approx_kl` generally "stays below 0.02"; higher values usually indicate a policy changing too quickly or a bug. Optional early stopping (`target-kl 0.01` in their code) is "an additional mechanism to explicitly enforce the trust-region constraint" — [Huang et al. 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)
- Andrychowicz et al. "suggest increasing N boosts the training throughput but makes the performance worse" (per sample) — [Huang et al. 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)

Andrychowicz et al. (more than 250,000 agents, more than 50 choices, five continuous-control tasks)
- Scale and scope of the study — [Andrychowicz et al. 2020, arXiv 2006.05990](https://arxiv.org/abs/2006.05990)
- "Use the PPO policy loss. Start with the clipping threshold set to 0.25 but also try lower and higher values if possible." — [Andrychowicz et al. 2020 (PDF)](https://arxiv.org/pdf/2006.05990v1)
- "Initialize the last policy layer with 100x smaller weights"; use tanh; use a separate, wide value network and tune the policy width — [Andrychowicz et al. 2020 (PDF)](https://arxiv.org/pdf/2006.05990v1)
- "Always use observation normalization and check if value function normalization improves performance"; gradient clipping is of secondary importance — [Andrychowicz et al. 2020 (PDF)](https://arxiv.org/pdf/2006.05990v1)
- "Use GAE with lambda = 0.9 but neither Huber loss nor PPO-style value loss clipping." — [Andrychowicz et al. 2020 (PDF)](https://arxiv.org/pdf/2006.05990v1)
- "Go over experience multiple times. Shuffle individual transitions before assigning them to minibatches and recompute advantages once per data pass." Many parallel envs and larger batches speed wall-clock but may hurt sample complexity — [Andrychowicz et al. 2020 (PDF)](https://arxiv.org/pdf/2006.05990v1)
- "Discount factor gamma is one of the most important hyperparameters and should be tuned per environment", starting from 0.99 — [Andrychowicz et al. 2020 (PDF)](https://arxiv.org/pdf/2006.05990v1)
- "Use Adam optimizer with momentum beta1 = 0.9 and a tuned learning rate (0.0003 is a safe default)"; linear decay helps slightly — [Andrychowicz et al. 2020 (PDF)](https://arxiv.org/pdf/2006.05990v1)
- No policy regularizer (entropy or KL-based) helped significantly in their tasks, except on HalfCheetah — [Andrychowicz et al. 2020 (PDF)](https://arxiv.org/pdf/2006.05990v1)

Engstrom et al.
- "Code-level optimizations", found only in implementations or described as auxiliary details, account for most of PPO's cumulative-reward advantage over TRPO and fundamentally change how the methods behave — [Engstrom et al. 2020, arXiv 2005.12729](https://arxiv.org/abs/2005.12729) (abstract only read)

SB3 specifics
- SB3 PPO defaults: `learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_kl=None` — [SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- The rollout buffer holds `n_steps * n_envs` samples; `clip_range_vf` "depends on the reward scaling" and is off when `None`; `target_kl` limits the KL between updates and is off by default — [SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- SB3 advice: do not assume defaults work; run several seeds because results vary with the seed; evaluate on a separate env over 5-20 episodes; try `deterministic=True` — [SB3 RL tips and tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)

Search with Optuna / RL Zoo
- Zoo command: `python train.py --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 --sampler tpe --pruner median`; default budget 500 trials; one pruning evaluation per 100k steps; studies can be stored with `--study-name` / `--storage` and viewed with `optuna-dashboard`; "not all hyperparameters are tuned"; search spaces live in `rl_zoo3/hyperparams_opt.py`; PPO tuning assumes `ortho_init = False` — [RL Zoo tuning docs](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html)
- Zoo PPO search space: `batch_size` = 2^k for k in 2..10; `n_steps` = 2^k for k in 5..12 (32-4096); `1 - gamma` log-uniform in [1e-4, 0.03]; `1 - gae_lambda` log-uniform in [1e-4, 0.1]; `learning_rate` log-uniform in [1e-5, 2e-3]; `ent_coef` log-uniform in [1e-8, 0.1]; `clip_range` in {0.1, 0.2, 0.3, 0.4}; `n_epochs` in {1, 5, 10, 20}; `max_grad_norm` uniform in [0.3, 2]; `net_arch` in {tiny = [64], small = [64, 64], medium = [256, 256]}; `activation_fn` in {tanh, relu} — [rl_zoo3/hyperparams_opt.py](https://raw.githubusercontent.com/DLR-RM/rl-baselines3-zoo/master/rl_zoo3/hyperparams_opt.py)
- The sampler table in that file has keys `a2c, ars, ddpg, dqn, qrdqn, ppo, ppo_lstm, sac, tqc, td3, trpo`; `ppo_mask` is not present — [rl_zoo3/hyperparams_opt.py](https://raw.githubusercontent.com/DLR-RM/rl-baselines3-zoo/master/rl_zoo3/hyperparams_opt.py)

### Inferences

Why gamma matters most here
- With reward only at hires and episodes of 100-300 agent steps, discounting per step is severe: 0.99^200 = 0.13, 0.995^200 = 0.37, 0.999^200 = 0.82. The task objective is undiscounted over a finite horizon, so gamma in {0.995, 0.999, 1.0} is the natural range, and 0.99 should be treated as a baseline to beat. This matches Andrychowicz et al.'s finding that gamma is among the most important choices and the Zoo's search down to 1 - gamma = 1e-4.

Suggested starting configuration (my recommendation, derived from the defaults and ranges cited above; not validated on the team's simulator)

| Parameter | Start | Search range | Reason |
|---|---|---|---|
| `n_envs` | 8-16 | fixed | cheap env; DummyVecEnv |
| `n_steps` | 512-1024 | 256-2048 | aim for `n_envs * n_steps` of 4,096-16,384, i.e. roughly 20-100 complete episodes per update, because returns are high-variance |
| `batch_size` | 256-512 | 64-1024 | larger minibatches average noisy advantages |
| `n_epochs` | 4-10 | {4, 5, 10} | watch `approx_kl` |
| `learning_rate` | 3e-4, linear decay | 1e-4 to 1e-3 (log) | Andrychowicz safe default |
| `gamma` | 0.999 | {0.99, 0.995, 0.999, 1.0} | see above |
| `gae_lambda` | 0.95 | 0.9-0.98 | 0.9 per Andrychowicz, 0.95 default |
| `ent_coef` | 0.01 | 1e-3 to 3e-2 (log) | discrete actions; SB3 default 0.0 risks early collapse |
| `clip_range` | 0.2 | {0.1, 0.2, 0.3} | |
| `vf_coef`, `max_grad_norm` | 0.5, 0.5 | fixed | secondary importance |
| `clip_range_vf` | `None` | fixed | value clipping does not help per both studies |
| network | 64x64 tanh, separate pi/vf | {64, 128, 256} width | separate value net per Andrychowicz |
| `target_kl` | `None` | optional 0.02-0.05 | safety net if KL spikes |

- Andrychowicz et al. studied continuous control only, so their numeric recommendations (and the finding that entropy regularization rarely helps) transfer imperfectly to a discrete, masked, stochastic task. Treat them as priors and confirm with ablations.
- Search procedure for a 5-week project: (1) fix the configuration above and get one learning run working; (2) one-at-a-time sweeps on gamma, `ent_coef`, learning rate and rollout size, 3 seeds each, which doubles as the required ablation section; (3) optionally 30-60 Optuna trials (TPE sampler, median pruner) at a reduced step budget, reusing the Zoo ranges above narrowed to the table; (4) retrain the best 2-3 configurations with 5 or more seeds at the full budget and report mean and spread. Tune on one set of seeds and report on different seeds.
- Since `ppo_mask` is absent from the Zoo's sampler table, write the Optuna objective by hand (create env, build `MaskablePPO` with sampled values, evaluate with the maskable `evaluate_policy`, report intermediate values for pruning). Copying the ranges from `sample_ppo_params` keeps the search defensible.
- Tune both policy variants with the same budget, otherwise the comparison between index-structured and unrestricted policies is confounded by tuning effort. The entity-based paper in Section 6 names exactly this confound as a limitation of its own results.

### Gaps
- No source gives hyperparameter ranges specifically for discrete, stochastic, sparse-terminal-reward tasks with 100-300 step episodes; the table is my synthesis.
- I could not confirm whether RL Zoo's `train.py` accepts `--algo ppo_mask` at all in 2.9.2; I only verified that no search space is defined for it.
- I could not confirm from the docs whether SB3's `ent_coef` accepts a schedule; assume a constant unless the team checks the source.
- Engstrom et al. was read at abstract level only; the list of individual code-level optimizations was not re-verified from the paper body.
- The last 1,662 characters of the Andrychowicz PDF were not read.

---

## 5. Sparse / delayed reward and high-variance returns: remedies that preserve the optimal policy

### Takeaway
Potential-based shaping F = gamma * Phi(s') - Phi(s) is the standard way to densify reward without changing optimal policies, and a natural potential here is an estimate of the value of the current pool. Scaling rewards by a positive constant (return normalization) and using larger batches are the other safe levers.

### Cited Findings
- Theorem (Ng, Harada & Russell): if F is potential-based, F(s, a, s') = gamma * Phi(s') - Phi(s), then every optimal policy of the shaped MDP is optimal in the original and vice versa — [University of Toronto CSC2542 slides on reward shaping](https://teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_reward_shaping.pdf) (secondary source summarising the paper)
- Under such shaping Q*_shaped(s, a) = Q*(s, a) - Phi(s) and V*_shaped(s) = V*(s) - Phi(s); the identities hold for any policy, so near-optimal policies are preserved too — [CSC2542 slides](https://teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_reward_shaping.pdf)
- "A way to define a good potential function might be to try to approximate V*(s)"; with Phi = V* the shaped value is identically zero — [CSC2542 slides](https://teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_reward_shaping.pdf)
- In the undiscounted case the result assumes Phi = 0 at the absorbing (terminal) state — [CSC2542 slides](https://teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_reward_shaping.pdf)
- Non-potential shaping can backfire: rewarding a bicycle agent for moving toward the goal produced an agent that "drove in circles with a radius of 20-50 meters around the starting point" (Randløv & Alstrøm 1998) — [CSC2542 slides](https://teach.cs.toronto.edu/~csc2542h/fall/material/csc2542f16_reward_shaping.pdf)
- PPO implementations for continuous control scale rewards by the standard deviation of a rolling discounted return and clip to [-10, 10]; SB3's `VecNormalize` exposes this as `norm_reward=True, clip_reward=10.0, gamma=0.99` — [Huang et al. 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details), [SB3 vectorized envs](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- SB3 normalizes advantages by default (`normalize_advantage=True`); the reference implementation does it per minibatch — [SB3 PPO docs](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [Huang et al. 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)
- Value-function normalization is worth testing per Andrychowicz et al. ("check if value function normalization improves performance") — [Andrychowicz et al. 2020 (PDF)](https://arxiv.org/pdf/2006.05990v1)

### Inferences
- A concrete potential for this simulator: Phi(s) = (expected value obtainable from the current pool under a simple heuristic), for example the sum of the top-k posterior means among present candidates, with k = roles left, times an acceptance probability. Set Phi(terminal) = 0 and use the same gamma in the shaping term as in PPO. With gamma < 1 the gamma factor must be included or invariance is lost.
- Implement shaping as a wrapper and always report the unshaped return in learning curves and evaluation. Keep the shaped and unshaped rewards in `info` so both can be logged. A shaping on/off run is a natural ablation.
- Reward normalization by a positive running scale preserves the optimal policy in principle, but `clip_reward` does not; check that clipping is rarely active (hire rewards scaled into roughly [-1, 1] by design make this moot).
- Simplest variance reducers that change nothing about the objective: more samples per update (larger `n_envs * n_steps`), larger minibatches, lower learning rate, and scaling rewards so that typical returns are of order 1 so the value loss is well conditioned.
- Interview costs already make the reward not purely terminal. If the quality of a hire is revealed only at the end, paying the posterior mean at offer acceptance instead of realised quality has the same expectation and lower variance; this is an expected-reward substitution, valid only if the posterior is the correct Bayesian one in the simulator.

### Gaps
- I did not retrieve the Ng, Harada & Russell paper itself; the theorem statement is from course slides, and the "ICML 1999" venue was not confirmed by any fetched page.
- No empirical source was found on how much potential-based shaping helps PPO specifically (as opposed to tabular methods); expect to establish this by ablation.
- Return-based scaling (PopArt and similar) was not researched.

---

## 6. Permutation-invariant / set-structured policies: Deep Sets, attention, padded slots

### Takeaway
The index-structured policy in Section 2 is already the minimal permutation-equivariant model (shared per-item network, no interaction). The next step up is Deep Sets style: add a masked pooled summary of all candidates to each candidate's input; attention is a further optional step, with evidence that it helps mainly when the entity set varies.

### Cited Findings
- Deep Sets studies models for "machine learning tasks defined on sets", considers "objective functions defined on sets that are invariant to permutations", gives a main theorem that "characterizes the permutation invariant functions", and derives conditions for permutation equivariance in deep models — [Zaheer et al. 2017, arXiv 1703.06114](https://arxiv.org/abs/1703.06114) (abstract only read)
- An entity-based PPO agent embeds each entity with a shared per-type embedding head, applies a 2-block Transformer with self-attention, and handles varying entity counts with ragged batches and masks; node selection uses a pointer-style query-key dot product with a softmax over eligible entities — [Symes Thompson et al. 2024, arXiv 2410.17647](https://arxiv.org/pdf/2410.17647)
- In that study, with a new topology every episode the MLP baseline "struggled and showed high variance" while the entity-based policy learned well; on a single fixed topology both converged and the gap shrank as the problem grew; entity policies generalised zero-shot across sizes (about 92.8 versus 93.8 mean reward) — [Symes Thompson et al. 2024](https://arxiv.org/pdf/2410.17647)
- Caveats stated by the authors: occasional training instability for the entity policy; different learning rates (0.005 entity versus 0.0003 MLP) and no hyperparameter sweeps for either agent; limited off-the-shelf tooling, harder to tune than MLPs — [Symes Thompson et al. 2024](https://arxiv.org/pdf/2410.17647)
- Tooling used there: Entity Gym, RogueNet, and enn-trainer (PPO adapted from CleanRL); the baseline was SB3 PPO with an MLP — [Symes Thompson et al. 2024](https://arxiv.org/pdf/2410.17647)

### Inferences
- Model ladder for the ablation, each step adding one capability: (a) index policy: logit_i = f(x_i, g); (b) Deep Sets equivariant: logit_i = f(x_i, g, pool_j phi(x_j)); (c) self-attention over candidates then a per-candidate head; (d) flat MLP on the concatenated observation (no symmetry). All four fit the `_build_mlp_extractor` hook from Section 2.
- Minimal Deep Sets layer: `h = phi(x)` with shape (B, N, d); `m = present` with shape (B, N, 1); `pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)`; feed `[x_i, g, pooled]` to the per-candidate head. For the "end day" logit and the critic, use `rho([g, pooled])`.
- Pitfalls with padded or empty slots, each of which I would unit-test:
  - Pooling without multiplying by the presence mask lets empty slots contribute `phi(0)`, which is not zero because of biases.
  - Mean pooling must divide by the number of present candidates, with a clamp for an empty pool; mean pooling also hides the pool size, so pass the count as a global feature or use sum plus mean.
  - Max pooling needs empty slots set to a large negative value first, and breaks when the pool is empty.
  - For attention, pass `key_padding_mask` for empty slots; a row with every key masked yields NaN in PyTorch's attention, so guard the empty-pool case.
  - Zero-filled features must be distinguishable from a real candidate whose features are zero: include an explicit `present` flag per slot (my toy env does).
  - Per-dimension observation normalization breaks weight sharing across slots (Section 1).
  - The flat-MLP variant is sensitive to slot ordering. Decide deliberately between a stable slot per candidate (identity persists, holes appear) and sorting by posterior mean (compact, but slots swap meaning between steps), keep it fixed across variants, and report it.
- The truncation to the top N candidates is itself part of the policy class. If the pool can exceed N, the selection rule (for example top N by posterior mean) can hide candidates from the agent; log how often the pool exceeds N.
- Expectation for results, based on the entity-based study: the structured policies should learn faster and more stably than the flat MLP when pool composition varies a lot, with the gap narrowing as training budget grows. That study's own confound (untuned, different learning rates) is a reminder to tune both arms equally.

### Gaps
- The Deep Sets decomposition (a set function is permutation invariant if and only if it can be written as rho of a sum of phi over elements) and the equivariant layer form are from the paper body, which I did not fetch; only the abstract was read. The implementation sketch above is standard practice rather than quoted text.
- No source was found comparing Deep Sets against attention inside PPO on a small CPU-scale task; the cyber-defence paper is the closest and uses a Transformer only.
- I did not test the Deep Sets or attention variants locally; only the index policy was run.

---

## 7. Logging and diagnostics: what SB3 logs, export, healthy curves, debugging checklist

### Takeaway
SB3 writes every curve the grading asks for (reward, value loss, entropy) to `progress.csv` once the logger is configured with the `csv` format, one row per PPO update. Entropy must be interpreted relative to the number of valid actions, and `explained_variance` and `approx_kl` are the two most informative health indicators.

### Cited Findings
- Logger setup: `from stable_baselines3.common.logger import configure`; `new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])`; `model.set_logger(new_logger)`. Available formats are `stdout, csv, log, tensorboard, json` — [SB3 logger docs](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)
- "When passing a custom logger object, this will overwrite `tensorboard_log` and `verbose` settings passed to the constructor." — [SB3 logger docs](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)
- `rollout/ep_rew_mean` and `rollout/ep_len_mean` are means over the last `stats_window_size` episodes (100 by default) of training episodes — [SB3 logger docs](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)
- `train/entropy_loss`: "Mean value of the entropy loss (negative of the average policy entropy)" — [SB3 logger docs](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)
- `train/explained_variance`: "Fraction of the return variance explained by the value function" (0 means no better than predicting zero, 1 is perfect, below 0 is worse than predicting zero) — [SB3 logger docs](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)
- `train/approx_kl`: "approximate mean KL divergence between old and new policy"; `train/clip_fraction`: mean fraction of the surrogate loss that was clipped; `train/value_loss`: error between value output and a Monte-Carlo or TD(lambda) estimate; `train/policy_gradient_loss`: its value "does not have much meaning" — [SB3 logger docs](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)
- `time/fps` includes time spent on gradient updates — [SB3 logger docs](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)
- CleanRL logs the equivalent set: `charts/episodic_return`, `charts/SPS`, `losses/value_loss`, `losses/policy_loss`, `losses/entropy`, `losses/approx_kl` (computed as `(logratio.exp() - 1) - logratio`), `losses/clipfrac`, `losses/explained_variance` — [CleanRL PPO docs](https://docs.cleanrl.dev/rl-algorithms/ppo/)
- Reference debug variables for PPO are policy_loss, value_loss, entropy_loss, clipfrac and approxkl, and `approx_kl` usually stays below 0.02 — [Huang et al. 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details)
- SB3's validation advice: get "signs of life" on toy problems, validate on progressively harder environments, watch tensor shapes because broadcast mistakes fail silently, and handle timeouts separately — [SB3 RL tips and tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- Since SB3 2.9.0, pandas is optional and needed only for loading results and plotting — [SB3 changelog](https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html)
- Local test: with `configure(path, ["csv"])`, `progress.csv` contained exactly these columns: `rollout/ep_len_mean, rollout/ep_rew_mean, time/fps, time/iterations, time/time_elapsed, time/total_timesteps, train/approx_kl, train/clip_fraction, train/clip_range, train/entropy_loss, train/explained_variance, train/learning_rate, train/loss, train/n_updates, train/policy_gradient_loss, train/value_loss`.
- Local test: with 21 actions, `train/entropy_loss` started near -1.47, far from -log(21) = -3.04, because few actions were valid in most states.

### Inferences
- Minimum logging for the report: `progress.csv` per run (one folder per variant and seed), plus a `MaskableEvalCallback` on a separate, fixed-seed evaluation env for deterministic-policy returns. Plot mean and spread across seeds against `time/total_timesteps`. Install `tensorboard` if the TensorBoard format is wanted; it is not pulled in by `pip install sb3-contrib` (it was absent in my environment), and I did not test TensorBoard output.
- `rollout/ep_rew_mean` is the stochastic training policy's return including any shaping, averaged over 100 episodes. Use the evaluation callback for the headline curve and report raw, unshaped return.
- Entropy curves: the maximum entropy in a state is log(number of valid actions), which changes with pool size and remaining interview slots. Add a custom callback that logs mean entropy divided by log(valid count), or the mean number of valid actions, so that a falling curve can be attributed to the policy rather than to the state distribution.
- What healthy curves look like (rules of thumb; only the `approx_kl` figure is sourced): evaluation return rising above the heuristic baselines and plateauing; `explained_variance` moving from about 0 toward clearly positive values (it will stay well below 1 here because outcomes are intrinsically noisy); `approx_kl` mostly below 0.02 without spikes; `clip_fraction` moderate rather than near 0 (updates too timid) or near 1 (updates too aggressive); entropy declining gradually rather than collapsing in the first few updates; `value_loss` falling after an initial rise as returns grow.
- Symptom to cause: entropy collapses early with flat reward → raise `ent_coef`, lower the learning rate, check the mask. `explained_variance` stuck at or below 0 → reward scale, gamma or the termination flag is wrong, or the observation lacks information such as days left. `approx_kl` spikes → fewer epochs, lower learning rate, or set `target_kl`. Reward improves in training but not in evaluation → masks or normalization statistics missing at evaluation.
- Debugging checklist, in order:
  1. Simulator unit tests: seeding reproducibility, posterior updates against closed form, departure and acceptance frequencies against their parameters.
  2. `check_env` on the single env (with invalid actions tolerated, see Section 1), then a random masked rollout asserting `mask.any()`, mask length 2N+1, and that every executed action was valid.
  3. Known-answer tests: a degenerate instance where the optimum is computable (for example one role, zero noise, no departures, so "offer the best visible candidate" is optimal) and PPO must reach it; a small instance solvable by dynamic programming as an upper bound.
  4. Baselines first: random valid policy and one or two heuristics (greedy on posterior mean, threshold rule), evaluated with the same seeds, to anchor the curves.
  5. Overfit a tiny instance: short horizon, N = 3, fixed seed set; PPO should beat the heuristics within minutes.
  6. Policy unit tests: permutation equivariance for variant (a); logits finite; masked actions have probability near zero.
  7. Scale up gradually (horizon, N, noise), re-checking curves at each step.
  8. Reproducibility: fix seeds, log library versions and the git commit, and run at least 3-5 seeds per configuration.

### Gaps
- No authoritative numeric thresholds were found for `clip_fraction` or `explained_variance`; the ranges above are heuristics.
- TensorBoard export and `MaskableEvalCallback` were not run locally (the maskable `evaluate_policy` was).
- I did not verify the Monitor CSV file format or the `load_results` helper in 2.9.x.

---

## 8. Compute budget: steps needed and wall-clock on a 4-8 core CPU

### Takeaway
On a cheap numpy env, SB3 MaskablePPO ran at roughly 4,000-7,000 environment steps per second on one core in my test, so a 1-million-step run takes about 2.5-4 minutes and a full tuning-plus-ablation campaign fits in hours, not days. How many steps the hiring task needs is not answered by any source and must be measured.

### Cited Findings
- Local test (toy env, 4-core sandbox, torch limited to 1 thread, `device="cpu"`): MaskablePPO `MlpPolicy` with `DummyVecEnv` x8 = about 7,000 steps/s; index policy with `DummyVecEnv` x8 = about 3,900 steps/s; `SubprocVecEnv` x4 = about 2,000 steps/s and x8 = about 3,000 steps/s. These include gradient updates, consistent with the SB3 definition of `time/fps` — [SB3 logger docs](https://stable-baselines3.readthedocs.io/en/master/common/logger.html)
- Local test: the toy problem showed clear learning within 40,000 steps (training return from -0.25 to 2.19; deterministic evaluation 3.01 ± 1.40 over 200 episodes), but its learned episodes were short (mean length about 18 steps), so it is easier than the real task.
- The Zoo's tuning example uses 50,000 steps per trial for MountainCar and a default of 500 trials — [RL Zoo tuning docs](https://rl-baselines3-zoo.readthedocs.io/en/master/guide/tuning.html)
- Reference PPO budgets in the μRTS masking study were 500,000 steps per run — [Huang & Ontañón 2020](https://arxiv.org/pdf/2006.14171v3)
- SB3: to improve performance, increase the training budget; no specific number is given — [SB3 RL tips and tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- More parallel envs raise throughput but can reduce per-sample efficiency — [Huang et al. 2022](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details), [Andrychowicz et al. 2020 (PDF)](https://arxiv.org/pdf/2006.05990v1)

### Inferences
- Planning numbers (to be replaced by the team's own measurement): at 4,000-7,000 steps/s, 1M steps takes about 2.5-4 minutes and 5M steps about 12-21 minutes per run on one core. With 100-300 steps per episode, 1M steps is only about 3,300-10,000 episodes, so several million steps per run is a plausible requirement for a noisy task; budget 1M-5M per run and check where curves plateau.
- Parallelise across runs, not within a run: launch one single-process training per core (different seeds or configurations), each with `torch.set_num_threads(1)`, rather than using `SubprocVecEnv`. On 4-8 cores this gives 4-8 runs concurrently.
- Example campaign on 8 cores: 4 policy variants x 5 seeds x 3M steps = 60M steps, about 2.5-4 hours of core time, under an hour of wall-clock; a 50-trial Optuna study at 500k steps per trial = 25M steps, about 1-2 hours of core time. Both are comfortably feasible inside the 5-week window, leaving room for ablations on gamma, entropy coefficient, shaping, masking versus penalty, and N.
- The team's batched simulator (1,000 episodes in 0.2 s) is far faster than SB3's per-step overhead, so the simulator is not the bottleneck; the Python loop over envs and the policy forward passes are. A custom batched `VecEnv` (Section 1) is the route if more throughput is needed.
- How to measure on the real env: run 100k steps, read `time/fps` from `progress.csv`, then extrapolate; repeat for each policy variant because attention models are slower.

### Gaps
- No published source was found for the number of steps PPO needs on small custom discrete tasks comparable to this one; all sample-budget figures here are extrapolations.
- Throughput was measured on a toy env in a 4-core sandbox and will differ for the team's simulator, larger N, and attention-based policies.
- I did not measure a custom batched `VecEnv` or any GPU configuration.
