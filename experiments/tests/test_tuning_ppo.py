"""PPO hyperparameter search: the sampled list, the step budgets and one tiny search run."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from hiring.training import load_run_config
from hiring.tuning_ppo import (CHOICES, FINAL_STEPS, LINEAR_RANGES, LOG_RANGES, PARAMETERS, apply, at_boundary,
                               sample_configs, search_steps, start_values)

BASE = load_run_config("ppo_stress")


def test_configuration_1_is_the_starting_values():
    first = sample_configs(BASE, 40)[0]
    assert first == start_values(BASE)
    assert first["gamma"] == 0.999 and first["learning_rate"] == 3e-4 and first["hidden"] == 64
    assert apply(BASE, first, curve=True) == BASE


def test_the_list_is_seeded_and_has_40_distinct_configurations():
    a, b = sample_configs(BASE, 40), sample_configs(BASE, 40)
    assert a == b and len(a) == 40
    assert len({json.dumps(c, sort_keys=True) for c in a}) == 40
    assert sample_configs(BASE, 40, search_seed=1)[1] != a[1]
    assert sample_configs(BASE, 10) == a[:10]          # a shorter list is a prefix of the longer one


def test_every_configuration_is_inside_the_plan_ranges():
    for c in sample_configs(BASE, 40)[1:]:
        assert set(c) == set(PARAMETERS)
        for k, v in CHOICES.items():
            assert c[k] in v
        for k, (lo, hi) in {**LOG_RANGES, **LINEAR_RANGES}.items():
            assert lo <= c[k] <= hi
        assert (c["n_steps"] * BASE["n_envs"]) % c["batch_size"] == 0


def test_search_length_is_the_same_share_of_each_final_length():
    assert FINAL_STEPS == {"index": 3_000_000, "unrestricted": 10_000_000, "pooled": 10_000_000}
    assert {a: search_steps(a) for a in FINAL_STEPS} == {"index": 900_000, "unrestricted": 3_000_000, "pooled": 3_000_000}


def test_apply_changes_only_the_searched_values():
    c = sample_configs(BASE, 40)[5]
    cfg = apply(BASE, c, curve=False)
    assert cfg["eval"]["episodes"] == 0 and BASE["eval"]["episodes"] == 300          # the base is not modified
    assert cfg["hidden"] == c["hidden"] and all(cfg["ppo"][k] == c[k] for k in c if k != "hidden")
    for k in ("vf_coef", "max_grad_norm", "lr_schedule"):
        assert cfg["ppo"][k] == BASE["ppo"][k]
    assert cfg["scenario_overrides"] == BASE["scenario_overrides"] and cfg["n_envs"] == BASE["n_envs"]


def test_boundary_report():
    c = dict(start_values(BASE))
    assert at_boundary(c) == ["n_epochs", "hidden"]     # 10 epochs and width 64 are ends of their lists
    c.update(learning_rate=1.05e-4, gae_lambda=0.979, gamma=1.0)
    assert set(at_boundary(c)) == {"n_epochs", "hidden", "learning_rate", "gae_lambda", "gamma"}


def test_one_tiny_search_run_is_scored_on_tuning_episodes_and_resumes(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "tune_ppo.py"
    spec = importlib.util.spec_from_file_location("tune_ppo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path)
    job = ("search", "index", 0, 101, 4096, 30)
    _, score = mod.job(job)
    out = mod.run_dir("search", "index", 0, 101)
    assert out.is_relative_to(tmp_path) and (out / "score.json").exists()
    assert score["configuration"] == 1 and score["params"] == start_values(BASE) and np.isfinite(score["mean"])
    assert len(np.load(out / "tuning_episode_payoffs.npy")) == 30
    assert json.loads((out / "run.json").read_text())["config"]["eval"]["episodes"] == 0
    stamp = (out / "model.zip").stat().st_mtime_ns
    _, again = mod.job(job)                              # a finished run is read back, not retrained
    assert again == score and (out / "model.zip").stat().st_mtime_ns == stamp


def test_frozen_final_configs_are_the_chosen_search_configurations():
    from hiring.training import run_config
    from hiring.tuning_ppo import CHOSEN
    configs = sample_configs(BASE, 40)
    for actor, c in CHOSEN.items():
        cfg = load_run_config(f"ppo_final_{actor}")
        want = apply(BASE, configs[c - 1], curve=True)
        want["total_timesteps"] = FINAL_STEPS[actor]
        assert cfg == want, actor
        small = run_config(f"ppo_final_{actor}", "small", "naive")
        assert small["ppo"] == cfg["ppo"] and small["hidden"] == cfg["hidden"]          # only the setting moves
        assert small["scenario"] == "small_exact" and small["scenario_overrides"] == {"obs_belief": "naive"}
        assert run_config(f"ppo_final_{actor}", "h0_0.05") == cfg                        # the file is the primary setting
