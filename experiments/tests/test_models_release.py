"""Packing the final models for the Release and verifying them after a download."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
from fetch_models import fetch  # noqa: E402
from pack_models import pack, read_checksums  # noqa: E402

SPEC = dict(actors=["index", "pooled"], blocks=[dict(setting="h0_0.05", belief="survival"), dict(setting="small", belief="survival")],
            ablations=dict(seeds=[1001], arms=[dict(name="gamma_0.99", actor="index")]))


def make_run(final, run, finished=True):
    (final / run).mkdir(parents=True)
    (final / run / "model_80.zip").write_bytes(f"80 {run}".encode())
    if finished:
        (final / run / "model.zip").write_bytes(f"model {run}".encode())
        (final / run / "run.json").write_text("{}")


@pytest.fixture
def final(tmp_path):
    final = tmp_path / "final"
    for folder, done in (("h0_0.05_survival", True), ("small_survival", False)):
        for actor in SPEC["actors"]:
            for seed in (1001, 1002):
                make_run(final, f"{folder}/{actor}_s{seed}", finished=done or seed == 1001)
        (final / folder / "seeds.json").write_text(json.dumps([1001, 1002]))
    return final


def test_only_finished_settings_are_packed_and_a_clean_clone_verifies(final, tmp_path):
    assert pack(final, SPEC) == ["models_h0_0.05_survival.zip"]          # small: one run unfinished; ablations: none
    sums = read_checksums(final)
    assert len(sums) == 2 * 2 * 2 + 1 and "release/models_h0_0.05_survival.zip" in sums
    assert pack(final, SPEC) == []                                       # nothing repacked
    clone = tmp_path / "clone"
    clone.mkdir()
    shutil.copy(final / "model_checksums.sha256", clone)
    assert fetch(clone, str(final / "release")) == []
    assert (clone / "h0_0.05_survival/pooled_s1002/model.zip").read_bytes() == b"model h0_0.05_survival/pooled_s1002"
    assert fetch(clone, "", verify_only=True) == []


def test_a_changed_or_missing_model_is_reported(final, tmp_path):
    pack(final, SPEC)
    clone = tmp_path / "clone"
    clone.mkdir()
    shutil.copy(final / "model_checksums.sha256", clone)
    fetch(clone, str(final / "release"))
    (clone / "h0_0.05_survival/index_s1001/model.zip").write_bytes(b"tampered")
    (clone / "h0_0.05_survival/index_s1002/model_80.zip").unlink()
    problems = fetch(clone, "", verify_only=True)
    assert len(problems) == 2 and any("differs" in p for p in problems) and any("missing" in p for p in problems)


def test_a_tampered_zip_is_not_unpacked(final, tmp_path):
    pack(final, SPEC)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "models_h0_0.05_survival.zip").write_bytes(b"not the zip")
    clone = tmp_path / "clone"
    clone.mkdir()
    shutil.copy(final / "model_checksums.sha256", clone)
    problems = fetch(clone, str(bad))
    assert any("not unpacked" in p for p in problems) and not (clone / "h0_0.05_survival").exists()


def test_a_model_changed_after_packing_stops_the_packer(final):
    pack(final, SPEC)
    (final / "h0_0.05_survival/index_s1001/model.zip").write_bytes(b"retrained")
    with pytest.raises(SystemExit):
        pack(final, SPEC)
