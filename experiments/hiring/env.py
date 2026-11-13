"""Hiring-race simulator as a Gymnasium environment with an action mask.

The task. Fill `roles` positions within `horizon` days. Candidates arrive with hidden quality
q ~ N(0, 1) and a noisy resume. An interview costs `interview_cost` and gives another noisy signal.
Each night a waiting candidate is poached with probability min(1, h0 * exp(slope * q)), so strong
candidates leave first. An offer is accepted with probability accept_base * exp(-accept_decay *
days waited); a candidate who declines leaves. Payoff = sum of hired q minus interview costs.

CONVENTIONS (fixed for gate G1; the exact solver and the bound must use the same ones)

  Day order   1. yesterday's interview results are applied to the estimates
              2. arrivals (with a pool cap, overflow is dropped automatically, see pool.py)
              3. offers
              4. interviews
              5. poaching (overnight)
  Phases      Strict. After the first interview of a day, offers are masked until the next day.
  Offers      One at a time. Allowed on the arrival day. Accept: a role is filled. Decline: the
              candidate leaves. One acceptance draw per candidate.
  Interviews  At most one per candidate per day, at most `interviews_per_day` per day, at most
              `max_interviews` per candidate. Masked on the last day (the result could never be used).
  Results     Visible next morning. An interview result is held as pending and applied at the
              start of the next day, before offers. If the candidate is poached that night the
              result is lost. So a candidate interviewed today can be offered tomorrow at the earliest.
  Episode end All roles filled, or the end of the last day. Both return terminated=True, never
              truncated=True: the deadline is part of the task and "days left" is observed.

ACTIONS     Discrete(2N + 1), N = n_slots.  i in [0, N): interview the candidate in slot i.
            N + i: offer to the candidate in slot i.  2N: end the day (always valid).

SLOTS       The N candidates with the highest NAIVE estimate, highest first, ties by arrival.
            The key does not depend on the belief variant, so every policy sees the same candidates.
            Order is fixed at the start of the day; candidates removed by offers close the gap.

OBSERVATION Scaled by hand, identically for every slot (no VecNormalize: it would break weight
            sharing). Per slot: present, estimate / 3, interviews seen / max_interviews,
            days waited / horizon, interviewed today. Binned scenarios add: resume bin and the
            count of interview outcomes per bin. Then five global features: days left / horizon,
            roles left / roles, interview slots left today / interviews_per_day, pool size / 2N
            (capped at 1), offers still open today. "estimate" is the survival-aware or the naive
            mean, chosen by scenario.obs_belief.

SEEDING     An episode seed fixes every random draw, indexed by candidate (seeding.py).
            reset(seed=s) plays episode s. reset() plays the next seed of `episode_seeds` if given,
            otherwise a fresh seed from the environment's own generator.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .belief import BinnedBelief, GaussianBelief, hazard
from .config import Scenario, load_seeds
from .pool import overflow_drop_index
from .seeding import EpisodeDraws, make_draws

N_GLOBAL = 5
MEAN_SCALE = 3.0


class HiringEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario: Scenario, strict: bool = True,
                 episode_seeds: Optional[Sequence[int]] = None):
        self.sc = sc = scenario
        self.strict = strict
        self.build_obs = True
        self.N = sc.n_slots
        self.binned = sc.signals == "binned"
        self.belief = BinnedBelief(sc) if self.binned else GaussianBelief(sc)
        self.n_slot_features = 5 + ((1 + self.belief.B) if self.binned else 0)
        self.action_space = spaces.Discrete(2 * self.N + 1)
        self.observation_space = spaces.Box(-5.0, 5.0, (self.N * self.n_slot_features + N_GLOBAL,), np.float32)
        self._episode_seeds = list(episode_seeds) if episode_seeds is not None else None
        self._seed_cursor = 0
        self._train_base = load_seeds()["training_episode_seed_start"]
        self.done = True

    # ---- reset ----------------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options or {}
        if "draws" in options:
            draws, self.episode_seed = options["draws"], None
        else:
            if "episode_seed" in options:
                self.episode_seed = int(options["episode_seed"])
            elif seed is not None:
                self.episode_seed = int(seed)
            elif self._episode_seeds is not None:
                self.episode_seed = int(self._episode_seeds[self._seed_cursor % len(self._episode_seeds)])
                self._seed_cursor += 1
            else:
                self.episode_seed = self._train_base + int(self.np_random.integers(0, 2 ** 40))
            draws = make_draws(self.episode_seed, self.sc)
        self._load(draws)
        return self._obs(), {"episode_seed": self.episode_seed}

    def _load(self, dr: EpisodeDraws):
        sc, n = self.sc, dr.n
        self.draws = dr
        self.q, self.arr = dr.quality, dr.arrival_day
        self.resume = dr.quality + sc.resume_sd * dr.resume_z
        self.haz = hazard(dr.quality, sc.h0, sc.quality_slope)
        self.alive = np.zeros(n, bool)
        self.n_obs = np.zeros(n, int)          # interview results already applied to the estimate
        self.n_done = np.zeros(n, int)         # interviews taken (includes today's pending one)
        self.isum = np.zeros(n)                # sum of applied interview signals (gaussian)
        self.has_pending = np.zeros(n, bool)
        self.pending = np.zeros(n)             # pending interview signal
        self.today = np.zeros(n, bool)         # interviewed today
        self.mu_naive = np.zeros(n)
        self.mu_surv = np.zeros(n)
        if self.binned:
            self.type = np.array([self.belief.fresh_type(self.belief.resume_bin(s)) for s in self.resume], int)
        self.t = 0
        self.next_arrival = 0
        self.roles_left = sc.roles
        self.done = False
        self.stats = dict(ret=0.0, interviews=0, hires=0, days=0, days_pool_over_slots=0,
                          days_hidden_better=0, hires_at_interview_cap=0, dropped_on_overflow=0)
        self._start_day()

    # ---- day mechanics --------------------------------------------------------------------------
    def _naive_of(self, idx):
        if self.binned:
            return self.belief.M[self.type[idx], 0]
        return self.belief.naive_mean(self.resume[idx], self.isum[idx], self.n_obs[idx])

    def _start_day(self):
        sc, t = self.sc, self.t
        # 1. yesterday's interview results become visible
        for c in np.flatnonzero(self.has_pending & self.alive):
            if self.binned:
                self.type[c] = self.belief.nxt[self.type[c], self.belief.interview_bin(self.pending[c])]
            else:
                self.isum[c] += self.pending[c]
            self.n_obs[c] += 1
        self.has_pending[:] = False
        self.today[:] = False
        # 2. arrivals, with automatic drop on overflow
        while self.next_arrival < len(self.arr) and self.arr[self.next_arrival] == t:
            self.alive[self.next_arrival] = True
            self.next_arrival += 1
            if sc.pool_cap is not None and self.alive.sum() > sc.pool_cap:
                idx = np.flatnonzero(self.alive)
                j = overflow_drop_index(self._naive_of(idx).tolist(), (t - self.arr[idx]).tolist(),
                                        self.type[idx].tolist() if self.binned else None)
                self.alive[idx[j]] = False
                self.stats["dropped_on_overflow"] += 1
        # 3. estimates and slot order for the day
        idx = np.flatnonzero(self.alive)
        d = t - self.arr[idx]
        if self.binned:
            mu = self.belief.M[self.type[idx], 0]
            ms = self.belief.M[self.type[idx], d]
        else:
            mu = self.belief.naive_mean(self.resume[idx], self.isum[idx], self.n_obs[idx])
            ms = self.belief.survival_mean(mu, self.n_obs[idx], d) if len(idx) else mu
        self.mu_naive[idx], self.mu_surv[idx] = mu, ms
        self.order = idx[np.lexsort((idx, -mu))].tolist()
        self.slots_left = sc.interviews_per_day
        self.offers_open = True
        # logging: does the N-slot view hide anything that matters?
        self.stats["days"] += 1
        if len(self.order) > self.N:
            self.stats["days_pool_over_slots"] += 1
            shown, hidden = self.order[:self.N], self.order[self.N:]
            if self.mu_surv[hidden].max() > self.mu_surv[shown].min():
                self.stats["days_hidden_better"] += 1

    def _poach(self):
        idx = np.flatnonzero(self.alive)
        gone = self.draws.u_poach[idx, self.t] < self.haz[idx]
        self.alive[idx[gone]] = False

    # ---- interface ------------------------------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        N = self.N
        mask = np.zeros(2 * N + 1, bool)
        mask[2 * N] = True                                   # "end day" is always valid
        v = self.order[:N]
        k = len(v)
        if k and not self.done:
            if self.offers_open:
                mask[N:N + k] = True
            if self.slots_left > 0 and self.t < self.sc.horizon - 1:
                mask[:k] = ~self.today[v] & (self.n_done[v] < self.sc.max_interviews)
        assert mask.any()
        return mask

    def step(self, action):
        assert not self.done, "episode is over; call reset()"
        action = int(action)
        if not self.action_masks()[action]:
            if self.strict:
                raise AssertionError(f"invalid action {action} on day {self.t}")
            return self._obs(), 0.0, False, False, {"invalid_action": True}
        sc, N = self.sc, self.N
        reward, terminated = 0.0, False
        if action == 2 * N:                                  # end day: poaching, then tomorrow
            self._poach()
            self.t += 1
            if self.t >= sc.horizon:
                terminated = True
            else:
                self._start_day()
        elif action < N:                                     # interview
            c = self.order[action]
            self.pending[c] = self.q[c] + sc.interview_sd * self.draws.interview_z[c, self.n_done[c]]
            self.has_pending[c] = True
            self.n_done[c] += 1
            self.today[c] = True
            self.slots_left -= 1
            self.offers_open = False
            self.stats["interviews"] += 1
            reward = -sc.interview_cost
        else:                                                # offer
            c = self.order.pop(action - N)
            self.alive[c] = False
            p = sc.accept_base * math.exp(-sc.accept_decay * (self.t - self.arr[c]))
            if self.draws.u_accept[c] < p:
                reward = float(self.q[c])
                self.roles_left -= 1
                self.stats["hires"] += 1
                self.stats["hires_at_interview_cap"] += int(self.n_done[c] >= sc.max_interviews)
                terminated = self.roles_left == 0
        self.stats["ret"] += reward
        info = {}
        if terminated:
            self.done = True
            info["episode_stats"] = dict(self.stats, episode_seed=self.episode_seed)
        return self._obs(), reward, terminated, False, info

    def _obs(self) -> np.ndarray:
        sc, N, F = self.sc, self.N, self.n_slot_features
        obs = np.zeros(N * F + N_GLOBAL, np.float32)
        if not self.build_obs:      # rule evaluation reads view() and does not need the vector
            return obs
        v = self.order[:N]
        k = len(v)
        if k:
            slots = np.zeros((k, F), np.float32)
            est = self.mu_surv[v] if sc.obs_belief == "survival" else self.mu_naive[v]
            slots[:, 0] = 1.0
            slots[:, 1] = est / MEAN_SCALE
            slots[:, 2] = self.n_obs[v] / sc.max_interviews
            slots[:, 3] = (self.t - self.arr[v]) / sc.horizon
            slots[:, 4] = self.today[v]
            if self.binned:
                b = self.belief
                for row, c in enumerate(v):
                    r, counts = b.types[self.type[c]]
                    slots[row, 5] = r / (b.R - 1)
                    slots[row, 6:] = np.asarray(counts) / sc.max_interviews
            obs[:k * F] = slots.ravel()
        g = N * F
        obs[g] = (sc.horizon - self.t) / sc.horizon
        obs[g + 1] = self.roles_left / sc.roles
        obs[g + 2] = self.slots_left / sc.interviews_per_day
        obs[g + 3] = min(len(self.order), 2 * N) / (2 * N)
        obs[g + 4] = float(self.offers_open)
        return obs

    def view(self) -> SimpleNamespace:
        """The same N-slot view as the observation, unscaled, for hand-written rules."""
        v = np.asarray(self.order[:self.N], int)
        return SimpleNamespace(
            ids=v, naive=self.mu_naive[v], survival=self.mu_surv[v], n_obs=self.n_obs[v],
            n_done=self.n_done[v], days=self.t - self.arr[v], today=self.today[v],
            t=self.t, horizon=self.sc.horizon, hired=self.sc.roles - self.roles_left,
            roles_left=self.roles_left, slots_left=self.slots_left, offers_open=self.offers_open,
            pool_size=len(self.order), types=self.type[v] if self.binned else None)
