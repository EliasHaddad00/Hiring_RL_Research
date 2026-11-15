"""Training script: a very short run of each actor writes every required log column.

These check the code path only (a few thousand steps), not learning.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from hiring.config import LockedPoolError, load_seeds
from hiring.evaluation import PPOPolicy, evaluate
from hiring.config import episode_seeds, load_scenario
from hiring.training import ACTORS, EVAL_POOL_OFFSET, REQUIRED_COLUMNS, load_run_config, train


def short_config(name):
    cfg = load_run_config(name)
    cfg["ppo"] = dict(cfg["ppo"], n_steps=64, batch_size=64, n_epochs=2)
    cfg["eval"] = dict(episodes=20, freq_steps=1024)
    return cfg


@pytest.fixture(scope="module", params=[(c, a) for c in ("ppo_small", "ppo_stress") for a in ACTORS],
                ids=lambda p: f"{p[0]}-{p[1]}")
def run(request, tmp_path_factory):
    name, actor = request.param
    out = tmp_path_factory.mktemp(f"{name}_{actor}")
    meta = train(short_config(name), actor, seed=101, out_dir=out, total_timesteps=2048)
    with open(out / "progress.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return name, actor, out, meta, rows


def test_all_six_columns_are_logged(run):
    _, _, _, _, rows = run
    for col in REQUIRED_COLUMNS:
        values = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
        assert values, f"{col} was never logged"
        assert np.isfinite(values).all(), f"{col} has non-finite values"


def test_normalized_entropy_is_logged_and_in_range(run):
    _, _, _, _, rows = run
    values = [float(r["train/normalized_entropy"]) for r in rows if r.get("train/normalized_entropy")]
    assert values and all(0.0 <= v <= 1.0 + 1e-6 for v in values)
    assert values[0] > 0.9          # a fresh policy is close to uniform over the valid actions


def test_evaluation_curve_uses_fixed_development_episodes_and_greedy_actions(run):
    _, _, out, meta, rows = run
    assert any(r.get("eval/mean_reward") for r in rows)
    ev = np.load(out / "evaluations.npz")
    assert ev["results"].shape[1] == 20 and len(ev["timesteps"]) >= 1
    dev = load_seeds()["episode_seeds"]["dev"]
    assert meta["eval_episode_seeds"][0] == dev["start"] + EVAL_POOL_OFFSET
    assert meta["eval_deterministic"] is True


def test_last_evaluation_matches_the_harness_on_the_same_episodes(run):
    """The curve and the final score measure the same thing: same episodes, masks, greedy actions."""
    name, _, out, meta, _ = run
    cfg = load_run_config(name)
    sc = load_scenario(cfg["scenario"], **(cfg.get("scenario_overrides") or {}))
    ev = np.load(out / "evaluations.npz")
    if ev["timesteps"][-1] != meta["total_timesteps"]:
        pytest.skip("the last evaluation was not at the final step")
    seeds = episode_seeds("dev", 20, offset=EVAL_POOL_OFFSET)
    ours = evaluate(sc, PPOPolicy.load(out / "model.zip"), seeds=seeds)["ret"]
    assert np.allclose(sorted(ours), sorted(ev["results"][-1]), atol=1e-5)


def test_run_metadata_and_saved_model(run):
    _, actor, out, meta, _ = run
    saved = json.loads((out / "run.json").read_text())
    assert saved["actor"] == actor and saved["seed"] == 101 and saved["total_timesteps"] >= 2048
    assert saved["config"]["ppo"]["gamma"] == 0.999
    assert (out / "model.zip").exists()


def test_final_seeds_are_refused(tmp_path):
    with pytest.raises(LockedPoolError):
        train(short_config("ppo_small"), "index", seed=1001, out_dir=tmp_path, total_timesteps=64)


def test_saved_model_holds_no_local_path(run):
    """Models are committed to a public repository. SB3 stores the learning-rate schedule inside
    model.zip; a lambda would be stored with the full path of training.py on the training machine."""
    import base64
    import json
    import zipfile
    from hiring import training

    out = run[2]
    data = json.loads(zipfile.ZipFile(Path(out) / "model.zip").read("data"))
    blobs = [base64.b64decode(v[":serialized:"]) for v in data.values() if isinstance(v, dict) and ":serialized:" in v]
    assert blobs
    here = str(Path(training.__file__).resolve().parents[2])
    for needle in (here, here.replace("\\", "/"), str(Path.home())):
        assert not any(needle.encode() in b for b in blobs), needle
