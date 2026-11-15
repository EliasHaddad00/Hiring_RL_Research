"""G2 gate check on the tuned rules, the bound and the first PPO runs.

Reads results/g2/tuning_40.json (best configuration per rule family, estimate and setting),
results/g2/bounds.json and, if present, PPO runs under results/g2/ppo_stress. Then, on
development episodes (not used for tuning):

  1. Bound minus every tuned rule at every setting, and minus every PPO run at h0 = 0.05.
     Pass: every difference positive. Differences under 2 standard errors are flagged.
  2. Slot view: each tuned rule through 10 slots, 20 slots and the full pool. Limit 0.02; above
     it the convention must go back to the user (CLAUDE.md).
  3. Share of hires at the 3-interview cap. Above 5% the user is told (CLAUDE.md).
  4. Shape facts: role term (bar lower with more open roles) and where the tuned linear bar would
     stand on the last day (it is not used there: the last-day bar is zero for every threshold rule).
  5. Calibrated scenario under its best rule: realised acceptance against 76 to 84%, share of
     roles filled, interviews per hire.

Writes results/g2/g2_check.{txt,json} and results/g2/calibrated_check.txt. Exit code 1 if the
bound is below any policy.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import PPOPolicy, evaluate, paired_compare, summarize  # noqa: E402
from hiring.rules import RULE_FAMILIES  # noqa: E402
from tune_rules import SETTINGS  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"
EPISODES, VIEW_EPISODES = 4000, 2000
VIEWS = {"10": 10, "20": 20, "full": 135}
VIEW_LIMIT, CAP_LIMIT = 0.02, 0.05
BOUND_LABEL = {"none": "stress, no pressure", "h0_0": "stress, h0 = 0 (decay only)", "h0_0.02": "stress, h0 = 0.02",
               "h0_0.05": "stress, h0 = 0.05 (primary)", "h0_0.10": "stress, h0 = 0.10", "calibrated": "calibrated"}


def scenario(setting, **extra):
    name, overrides = SETTINGS[setting]
    return load_scenario(name, **dict(overrides, **extra))


def rule_job(args):
    setting, family, belief, params, view = args
    n = EPISODES if view == "10" else VIEW_EPISODES
    res = evaluate(scenario(setting, n_slots=VIEWS[view]), RULE_FAMILIES[family](params, belief=belief),
                   seeds=episode_seeds("dev", n))
    return (setting, family, belief, view), res


def ppo_job(path):
    import torch as th
    th.set_num_threads(1)
    meta = json.loads((Path(path) / "run.json").read_text())
    sc = load_scenario(meta["config"]["scenario"], **(meta["config"].get("scenario_overrides") or {}))
    res = evaluate(sc, PPOPolicy.load(Path(path) / "model.zip"), seeds=episode_seeds("dev", EPISODES))
    return Path(path).name, meta, summarize(res)


def main():
    tuned = json.loads((OUT / "tuning_40.json").read_text())
    bounds = {r["setting"]: r["bound"] for r in json.loads((OUT / "bounds.json").read_text())["settings"]}
    jobs = [(t["setting"], t["family"], t["belief"], t["params"], v) for t in tuned for v in VIEWS]
    ppo_runs = sorted(str(p.parent) for p in (OUT / "ppo_stress").glob("*/model.zip"))
    with Pool(14) as pool:
        raw = dict(pool.map(rule_job, jobs, chunksize=1))
        ppo = pool.map(ppo_job, ppo_runs, chunksize=1)

    problems, flags, rows = [], [], []
    lines = ["1. Bound against every tuned rule (4,000 development episodes; tuning score beside it)", ""]
    settings = list(dict.fromkeys(t["setting"] for t in tuned))
    for s in settings:
        bound = bounds.get(BOUND_LABEL[s])
        lines.append(f"{s}: bound {bound:.4f}" if bound is not None else f"{s}: bound not computed yet")
        lines.append(f"  {'family':22s} {'estimate':9s} {'tuning':>7s} {'dev':>7s} {'se':>6s} {'bound - dev':>12s} {'% of bound':>11s} "
                     f"{'full - 10 slots':>16s} {'20 - 10':>8s} {'at cap':>7s}")
        for t in (t for t in tuned if t["setting"] == s):
            key = (s, t["family"], t["belief"])
            sm = summarize(raw[key + ("10",)])
            ten = raw[key + ("10",)]["ret"][:VIEW_EPISODES]
            gap_full = paired_compare(raw[key + ("full",)]["ret"], ten)
            gap_20 = paired_compare(raw[key + ("20",)]["ret"], ten)
            margin = None if bound is None else bound - sm["mean"]
            rows.append(dict(setting=s, family=t["family"], belief=t["belief"], params=t["params"], tuning=t["best"],
                             dev=sm, bound=bound, bound_minus_dev=margin, full_minus_10=gap_full, slots20_minus_10=gap_20))
            lines.append(f"  {t['family']:22s} {t['belief']:9s} {t['best']:7.3f} {sm['mean']:7.3f} {sm['se']:6.3f} "
                         + (f"{margin:+12.3f} {100 * sm['mean'] / bound:10.1f}%" if bound is not None else f"{'':>12s} {'':>11s}")
                         + f" {gap_full['diff']:+16.4f} {gap_20['diff']:+8.4f} {100 * sm['share_hires_at_interview_cap']:6.1f}%")
            name = f"{s} / {t['family']} / {t['belief']}"
            if margin is not None and margin <= 0:
                problems.append(f"bound below rule: {name} ({sm['mean']:.3f} against {bound:.3f})")
            elif margin is not None and margin < 2 * sm["se"]:
                flags.append(f"bound within 2 standard errors of {name}")
            if abs(gap_full["diff"]) > VIEW_LIMIT:
                flags.append(f"SLOT VIEW: {name}: full pool minus 10 slots = {gap_full['diff']:+.4f} (limit {VIEW_LIMIT})")
            if sm["share_hires_at_interview_cap"] > CAP_LIMIT:
                flags.append(f"interview cap: {name}: {100 * sm['share_hires_at_interview_cap']:.1f}% of hires at the cap of 3")
        best = max((r for r in rows if r["setting"] == s), key=lambda r: r["dev"]["mean"])
        lines.append(f"  best on development episodes: {best['family']} ({best['belief']}), {best['dev']['mean']:.3f}"
                     + (f", {100 * best['dev']['mean'] / bound:.1f}% of the bound" if bound is not None else ""))
        lines.append("")

    lines += ["2. PPO runs at h0 = 0.05 against the bound (greedy actions, masks on, 4,000 development episodes)", ""]
    ppo_rows = []
    b05 = bounds.get(BOUND_LABEL["h0_0.05"])
    for name, meta, sm in ppo:
        ppo_rows.append(dict(run=name, actor=meta["actor"], seed=meta["seed"], steps=meta["total_timesteps"], dev=sm))
        lines.append(f"  {name:22s} {meta['total_timesteps']:9d} steps  {sm['mean']:.3f} +- {sm['se']:.3f}   bound - PPO {b05 - sm['mean']:+.3f}"
                     f"   interviews per episode {sm['interviews_per_episode']:.1f}, hires {sm['hires_per_episode']:.2f}, "
                     f"hires at the interview cap {100 * sm['share_hires_at_interview_cap']:.1f}%")
        if sm["share_hires_at_interview_cap"] > CAP_LIMIT:
            flags.append(f"interview cap: PPO run {name}: {100 * sm['share_hires_at_interview_cap']:.1f}% of hires at the cap of 3")
        if b05 - sm["mean"] <= 0:
            problems.append(f"bound below PPO run {name}")
    if not ppo:
        lines.append("  no PPO runs under results/g2/ppo_stress yet")

    lines += ["", "3. Shape facts for the tuned threshold rules", ""]
    for r in rows:
        if r["family"] in ("resume_falling_bar", "one_interview", "up_to_two_interviews"):
            p = r["params"]
            role = p[-1]
            bar0, fall0 = (p[0], p[1]) if r["family"] == "resume_falling_bar" else \
                ((p[0], p[2]) if r["family"] == "one_interview" else (p[0], p[3]))
            f_last = 59 / 60
            r["role_term"], r["last_day_bar_no_hires"] = role, bar0 - fall0 * f_last
            lines.append(f"  {r['setting']:11s} {r['family']:22s} {r['belief']:9s} role term {role:+.3f} "
                         f"({'bar lower with more open roles' if role > 0 else 'NOT lower with more open roles'}); "
                         f"tuned linear bar on the last day {bar0 - fall0 * f_last:+.3f} (not used: the last-day bar is zero)")

    lines += ["", "Flags: " + ("none" if not flags else "")] + [f"  {f}" for f in flags]
    lines += ["Problems: " + ("none" if not problems else "")] + [f"  {p}" for p in problems]
    text = "\n".join(lines)
    print(text)
    (OUT / "g2_check.txt").write_text(text + "\n")
    (OUT / "g2_check.json").write_text(json.dumps(dict(rules=rows, ppo=ppo_rows, flags=flags, problems=problems), indent=1))

    cal = [t for t in tuned if t["setting"] == "calibrated"]
    if cal:
        top = max(cal, key=lambda t: t["best"])
        sm = top["summary"]
        acc = sm["realised_acceptance"]
        out = [f"Calibrated scenario, best tuned rule on the 4,000 tuning episodes: {top['family']} ({top['belief']}), {top['best']:.3f}",
               f"  realised acceptance  {100 * acc:.1f}%   benchmark 76 to 84%: {'inside' if 0.76 <= acc <= 0.84 else 'OUTSIDE'}",
               f"  share of roles filled {100 * sm['hires_per_episode'] / 3:.1f}%   (single vacancies: 84 to 93% within 60 days, constant-hazard reading)",
               f"  interviews per hire  {sm['interviews_per_episode'] / sm['hires_per_episode']:.1f}   benchmark about 20",
               f"  share of hires at the 3-interview cap {100 * sm['share_hires_at_interview_cap']:.1f}%", "",
               "All rules on the tuning episodes (realised acceptance):"]
        out += [f"  {t['family']:22s} {t['belief']:9s} {t['best']:.3f}  {100 * t['summary']['realised_acceptance']:.1f}%" for t in cal]
        print("\n" + "\n".join(out))
        (OUT / "calibrated_check.txt").write_text("\n".join(out) + "\n")
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
