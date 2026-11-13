"""G1: the earlier rule scores reproduce in our environment (slow).

The earlier simulator's own evaluation draws are replayed through our environment with its
full-precision tuned parameters (results/g1/v3_fullprec.json, written by the reference copy
scripts/reference/hiring_sim_v3_fullprec.py; notes/hiring_sim_v3.py itself is untouched).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts" / "reference"))

from reproduce_rules import GATE_ROWS, PUBLISHED, TOLERANCE, ours, v3_reference  # noqa: E402
from v3_bridge import SETTINGS, v3, v3_episode_draws, v3_key  # noqa: E402

from hiring.config import load_scenario  # noqa: E402
from hiring.evaluation import evaluate  # noqa: E402
from hiring.rules import FAMILIES  # noqa: E402

pytestmark = pytest.mark.slow
PARAMS = json.loads((ROOT / "results" / "g1" / "v3_fullprec.json").read_text())
LEVEL = {label: (h0, b) for label, h0, b in SETTINGS}


def test_full_precision_rerun_matches_the_original_results_file():
    """The rerun must print the same rounded parameters and scores as notes/hiring_sim_v3_results.txt."""
    original = (ROOT / "notes" / "hiring_sim_v3_results.txt").read_text(encoding="utf-8")
    checked = 0
    for label, h0, b in SETTINGS:
        block = original.split(f"== h0={h0} b={b} ")[1].split("==")[0]
        for name, row in PARAMS[v3_key(h0, b)].items():
            line = next(l for l in block.splitlines() if l.strip().startswith(name + " "))
            assert f"par={row['par']}" in line, (label, name, line, row["par"])
            assert f" {row['mean']:.2f} " in line, (label, name, line, row["mean"])
            if "par_full" in row:
                assert np.allclose(np.round(row["par_full"], 2), row["par"])
            checked += 1
    assert checked == 40


@pytest.mark.parametrize("label,family", GATE_ROWS)
def test_gate_rows_reproduce_within_tolerance(label, family):
    """10-slot view, v3 offer details, all 4,000 of v3's evaluation episodes."""
    h0, b = LEVEL[label]
    par = PARAMS[v3_key(h0, b)][family]["par_full"]
    mean = ours(h0, b, family, par, "view10_compat")["ret"].mean()
    assert abs(mean - PUBLISHED[label][family]) <= TOLERANCE, (mean, PUBLISHED[label][family])


@pytest.mark.parametrize("label,family", [("none", "up_to_two_interviews"), ("h0_0.02", "one_interview"),
                                          ("h0_0.05", "up_to_two_interviews"), ("h0_0.10", "resume_falling_bar")])
def test_episode_by_episode_match_with_full_pool(label, family):
    """With the full pool visible and v3's offer details, every episode must give v3's payoff,
    plus 0.05 for each last-day interview v3 took (we mask those)."""
    h0, b = LEVEL[label]
    par = PARAMS[v3_key(h0, b)][family]["par_full"]
    d = v3.gen(1000, 100)
    from v3_bridge import v3_rule_kwargs
    val, nint, nlast = v3.run(d, h0, b, detail=True, **v3_rule_kwargs(family, par))
    sc = load_scenario("stress", h0=h0, accept_decay=b, n_slots=135)
    res = evaluate(sc, FAMILIES[family](par, v3_compat=True), draws_list=[v3_episode_draws(d, e) for e in range(1000)])
    assert np.allclose(res["ret"], val + 0.05 * nlast, atol=1e-9)
    assert np.array_equal(res["interviews"], nint - nlast)
