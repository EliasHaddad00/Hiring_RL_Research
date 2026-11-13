"""G1 check: reproduce the earlier rule scores in our environment.

For each pressure level and rule family, with v3's full-precision tuned parameters
(results/g1/v3_fullprec.json) and v3's own 4,000 evaluation episodes replayed draw for draw:

  v3              the earlier simulator itself (reference copy)
  full_compat     our environment, full pool visible, v3 offer details on
  view10_compat   our environment, 10-slot view, v3 offer details on      <- the G1 comparison
  view10_default  our environment, 10-slot view, default rule behaviour
  full_default    our environment, full pool visible, default rule behaviour

Writes results/g1/rule_reproduction.{json,txt,npz}.  Usage: python reproduce_rules.py [processes]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from v3_bridge import EVAL_SEEDS, SETTINGS, v3, v3_episode_draws, v3_key, v3_rule_kwargs  # noqa: E402

from hiring.config import load_scenario  # noqa: E402
from hiring.evaluation import evaluate, summarize  # noqa: E402
from hiring.rules import FAMILIES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "g1"
FULL_POOL = 135
VARIANTS = {"full_compat": (FULL_POOL, True), "view10_compat": (10, True),
            "view10_default": (10, False), "full_default": (FULL_POOL, False)}

# Scores printed in notes/hiring_sim_v3_results.txt (4,000 episodes, standard error about 0.02)
PUBLISHED = {
    "none":       {"static_threshold": 3.10, "resume_falling_bar": 3.42, "one_interview": 3.86, "up_to_two_interviews": 3.93},
    "decay_only": {"static_threshold": 3.10, "resume_falling_bar": 3.31, "one_interview": 3.71, "up_to_two_interviews": 3.71},
    "h0_0.02":    {"static_threshold": 3.10, "resume_falling_bar": 3.17, "one_interview": 3.54, "up_to_two_interviews": 3.54},
    "h0_0.05":    {"static_threshold": 3.10, "resume_falling_bar": 3.15, "one_interview": 3.34, "up_to_two_interviews": 3.42},
    "h0_0.10":    {"static_threshold": 3.10, "resume_falling_bar": 3.17, "one_interview": 3.14, "up_to_two_interviews": 3.14},
}
# The rows gate G1 names: fixed threshold everywhere, and the best interview rule at each level.
GATE_ROWS = [(s, "static_threshold") for s, _, _ in SETTINGS] + [
    ("none", "up_to_two_interviews"), ("decay_only", "one_interview"), ("h0_0.02", "one_interview"),
    ("h0_0.05", "up_to_two_interviews"), ("h0_0.10", "one_interview")]
TOLERANCE = 0.05
VIEW_TOLERANCE = 0.02


def load_params():
    return json.loads((OUT / "v3_fullprec.json").read_text())


def scenario(h0, b, n_slots):
    return load_scenario("stress", h0=h0, accept_decay=b, n_slots=n_slots, max_candidates=FULL_POOL)


def v3_reference(h0, b, family, par):
    """Per-episode payoff, interviews and last-day interviews from the earlier simulator."""
    parts = [v3.run(v3.gen(1000, s), h0, b, detail=True, **v3_rule_kwargs(family, par)) for s in EVAL_SEEDS]
    return [np.concatenate(x) for x in zip(*parts)]


def ours(h0, b, family, par, variant):
    n_slots, compat = VARIANTS[variant]
    draws = [v3_episode_draws(d, e) for d in (v3.gen(1000, s) for s in EVAL_SEEDS) for e in range(1000)]
    return evaluate(scenario(h0, b, n_slots), FAMILIES[family](par, v3_compat=compat), draws_list=draws)


def job(args):
    label, h0, b, family, par, variant = args
    if variant == "v3":
        val, nint, nlast = v3_reference(h0, b, family, par)
        return label, family, variant, dict(ret=val, interviews=nint, last_day_interviews=nlast)
    return label, family, variant, ours(h0, b, family, par, variant)


def main():
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    params = load_params()
    jobs = []
    for label, h0, b in SETTINGS:
        for family in FAMILIES:
            par = params[v3_key(h0, b)][family]["par_full"]
            jobs += [(label, h0, b, family, par, v) for v in ["v3", *VARIANTS]]
    t0 = time.time()
    with Pool(procs) as pool:
        raw = {(lab, fam, var): res for lab, fam, var, res in pool.imap_unordered(job, jobs)}

    report, lines, stop = {}, [], []
    episodes = {}
    for label, h0, b in SETTINGS:
        for family in FAMILIES:
            ref = raw[(label, family, "v3")]
            row = dict(published=PUBLISHED[label][family], par_full=params[v3_key(h0, b)][family]["par_full"],
                       v3_mean=float(ref["ret"].mean()),
                       v3_last_day_interviews_per_episode=float(ref["last_day_interviews"].mean()),
                       v3_last_day_cost_per_episode=float(0.05 * ref["last_day_interviews"].mean()))
            for var in VARIANTS:
                res = raw[(label, family, var)]
                s = summarize(res)
                s["mean_minus_published"] = s["mean"] - row["published"]
                s["paired_diff_vs_v3"] = float((res["ret"] - ref["ret"]).mean())
                row[var] = s
                episodes[f"{label}|{family}|{var}"] = res["ret"]
            episodes[f"{label}|{family}|v3"] = ref["ret"]
            # exactness: with the full pool and v3's offer details, the only difference left is that
            # we mask last-day interviews, worth exactly 0.05 each
            resid = raw[(label, family, "full_compat")]["ret"] - (ref["ret"] + 0.05 * ref["last_day_interviews"])
            row["full_compat_max_abs_residual"] = float(np.abs(resid).max())
            row["episodes_differing"] = int((np.abs(resid) > 1e-9).sum())
            row["view_gap_compat"] = row["view10_compat"]["mean"] - row["full_compat"]["mean"]
            row["view_gap_default"] = row["view10_default"]["mean"] - row["full_default"]["mean"]
            row["gate_row"] = (label, family) in GATE_ROWS
            row["within_tolerance"] = abs(row["view10_compat"]["mean_minus_published"]) <= TOLERANCE
            p = row["par_full"]
            if family != "static_threshold":
                r_term = p[{"resume_falling_bar": 2, "one_interview": 6, "up_to_two_interviews": 8}[family]]
                rule = FAMILIES[family](p)
                row["shape_bar_lower_with_more_open_roles"] = bool(r_term > 0)
                row["shape_last_day_resume_bar"] = float(rule.offer_bar(0, 59 / 60, 0))
            if max(abs(row["view_gap_compat"]), abs(row["view_gap_default"])) > VIEW_TOLERANCE:
                stop.append(f"{label}/{family}: 10-slot view vs full pool differ by more than {VIEW_TOLERANCE}")
            report.setdefault(label, {})[family] = row

    hdr = (f"{'setting':11s} {'rule':22s} {'publ.':>5s} {'v3':>6s} {'full+c':>6s} {'v10+c':>6s} {'v10':>6s} {'full':>6s} "
           f"{'se':>5s} {'lastday/ep':>10s} {'resid':>8s} {'pool>10':>7s} {'hidden>':>7s} {'at cap':>6s} {'int/ep':>6s} gate")
    lines.append(hdr)
    for label, _, _ in SETTINGS:
        for family in FAMILIES:
            r = report[label][family]
            v = r["view10_compat"]
            flag = ("PASS" if r["within_tolerance"] else "FAIL") if r["gate_row"] else "-"
            lines.append(
                f"{label:11s} {family:22s} {r['published']:5.2f} {r['v3_mean']:6.3f} {r['full_compat']['mean']:6.3f} "
                f"{v['mean']:6.3f} {r['view10_default']['mean']:6.3f} {r['full_default']['mean']:6.3f} {v['se']:5.3f} "
                f"{r['v3_last_day_interviews_per_episode']:10.4f} {r['full_compat_max_abs_residual']:8.1e} "
                f"{v['share_days_pool_over_slots']:7.3f} {v['share_days_hidden_better']:7.4f} "
                f"{v['share_hires_at_interview_cap']:6.3f} {v['interviews_per_episode']:6.1f} {flag}")
    gate_ok = all(report[s][f]["within_tolerance"] for s, f in GATE_ROWS)
    lines.append("")
    lines.append(f"Gate rows within {TOLERANCE} of the published score (10-slot view, v3 offer details): "
                 f"{'ALL PASS' if gate_ok else 'NOT ALL'}")
    lines.append("Columns: full+c / v10+c = full pool / 10-slot view with v3 offer details; v10 / full = default rule. "
                 "resid = largest per-episode gap between full+c and v3 after adding back 0.05 per v3 last-day interview.")
    for s in stop:
        lines.append("STOP: " + s)
    lines.append(f"({time.time() - t0:.0f}s)")
    text = "\n".join(lines)
    print(text)
    (OUT / "rule_reproduction.txt").write_text(text + "\n")
    (OUT / "rule_reproduction.json").write_text(json.dumps(dict(report=report, gate_pass=gate_ok, stop=stop), indent=1))
    np.savez_compressed(OUT / "rule_reproduction_episodes.npz", **episodes)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
