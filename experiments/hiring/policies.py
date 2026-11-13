"""Policy networks for MaskablePPO.

Unrestricted agent: MaskablePPO("MlpPolicy"), separate 64 x 64 actor and critic. No code needed.

Index-structured agent (IndexPolicy): one shared network scores each candidate from that
candidate's own features plus the global features, and outputs two logits per candidate
(interview, offer). A second small network gives the "end day" logit from the global features
only (a stated restriction). The actor therefore cannot let one candidate's score depend on
another candidate. The critic is the same unrestricted 64 x 64 network as in the other agent,
so the two agents differ only in the actor.

The extractor emits the 2N + 1 logits directly, in the environment's action order
(interview slots, offer slots, end day), and SB3's final `action_net` is replaced by
nn.Identity(). Leaving SB3's default dense output layer in place would mix candidates and
silently destroy the index structure.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import torch as th
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn


def mlp(sizes, activation):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), activation()]
    return nn.Sequential(*layers)


class IndexExtractor(nn.Module):
    def __init__(self, n_slots, n_slot_features, n_global, hidden=64, activation=nn.Tanh):
        super().__init__()
        self.N, self.F, self.G = n_slots, n_slot_features, n_global
        self.candidate_body = mlp([self.F + self.G, hidden, hidden], activation)
        self.candidate_head = nn.Linear(hidden, 2)          # interview logit, offer logit
        self.end_body = mlp([self.G, hidden], activation)
        self.end_head = nn.Linear(hidden, 1)                # "end day" logit, from globals only
        self.critic = mlp([self.N * self.F + self.G, hidden, hidden], activation)
        self.latent_dim_pi = 2 * self.N + 1
        self.latent_dim_vf = hidden

    def forward_actor(self, x: th.Tensor) -> th.Tensor:
        b = x.shape[0]
        slots = x[:, :self.N * self.F].reshape(b, self.N, self.F)
        g = x[:, self.N * self.F:]
        per = self.candidate_head(self.candidate_body(th.cat([slots, g[:, None, :].expand(b, self.N, self.G)], dim=-1)))
        end = self.end_head(self.end_body(g))
        return th.cat([per[:, :, 0], per[:, :, 1], end], dim=1)

    def forward_critic(self, x: th.Tensor) -> th.Tensor:
        return self.critic(x)

    def forward(self, x: th.Tensor):
        return self.forward_actor(x), self.forward_critic(x)


class IndexPolicy(MaskableActorCriticPolicy):
    """policy_kwargs must give n_slots, n_slot_features and n_global (from the environment)."""

    def __init__(self, *args, n_slots: int, n_slot_features: int, n_global: int, hidden: int = 64, **kwargs):
        self._index_spec = dict(n_slots=n_slots, n_slot_features=n_slot_features, n_global=n_global, hidden=hidden)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = IndexExtractor(activation=self.activation_fn, **self._index_spec)

    def _build(self, lr_schedule) -> None:
        self._build_mlp_extractor()
        assert self.features_dim == self.mlp_extractor.N * self.mlp_extractor.F + self.mlp_extractor.G
        assert self.action_space.n == self.mlp_extractor.latent_dim_pi
        self.action_net = nn.Identity()                     # the extractor already outputs the logits
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            self.mlp_extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))
            for head in (self.mlp_extractor.candidate_head, self.mlp_extractor.end_head):
                head.apply(partial(self.init_weights, gain=0.01))
            self.value_net.apply(partial(self.init_weights, gain=1))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(self._index_spec)
        return data


def index_policy_kwargs(env) -> dict:
    """policy_kwargs for IndexPolicy matching a HiringEnv's observation layout."""
    from .env import N_GLOBAL
    return dict(n_slots=env.N, n_slot_features=env.n_slot_features, n_global=N_GLOBAL)
