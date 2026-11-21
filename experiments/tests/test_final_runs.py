"""The final run list: order, size and the pre-registered cutoff rule."""
import datetime
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
from final_runs import block_seeds, expand  # noqa: E402

SPEC = yaml.safe_load((ROOT / "experiments" / "configs" / "final_runs.yaml").read_text())


def test_run_list_is_360_runs_with_primary_then_h4b_first():
    assert sum(len(r["seeds"]) for r in expand(SPEC)) == 360
    assert [b["group"] for b in SPEC["blocks"]][:3] == ["primary", "belief arm (H4b)", "other settings"]
    assert SPEC["seeds"] == list(range(1001, 1016)) and SPEC["procs"] == 6


def test_cutoff_touches_only_other_settings_not_yet_started(tmp_path):
    before, after = datetime.date(2026, 12, 4), datetime.date(2026, 12, 5)
    primary, h4b, other = SPEC["blocks"][0], SPEC["blocks"][1], SPEC["blocks"][2]
    assert block_seeds(SPEC, primary, tmp_path / "p", after) == SPEC["seeds"]
    assert block_seeds(SPEC, h4b, tmp_path / "h", after) == SPEC["seeds"]
    assert block_seeds(SPEC, other, tmp_path / "late", after) == list(range(1001, 1011))
    assert block_seeds(SPEC, other, tmp_path / "early", before) == SPEC["seeds"]
    assert block_seeds(SPEC, other, tmp_path / "early", after) == SPEC["seeds"]    # started in time: stays at 15
    assert block_seeds(SPEC, other, tmp_path / "late", before) == list(range(1001, 1011))
