"""Gate G4, step 2: apply the pre-registered decision rule and name one verdict per hypothesis.

  python experiments/scripts/apply_decision_rule.py              the final results (results/final)
  python experiments/scripts/apply_decision_rule.py --stand-in   pipeline test, NOT results

Reads only files written by score_final.py and by earlier gates; plays no episode. Writes
verdicts.txt and verdicts.json into the folder it read from.

The rule is PREREGISTRATION.md, "Decision rule", implemented in hiring/decision.py. Where that
text leaves a choice open, the choice is read from configs/decision_rule.yaml. On the final
results the script refuses to run while any of them is undecided, and while any registered run
is not scored (no seed is dropped). --stand-in uses the stand-in values, which are not decisions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_plateau import boot_interval as plateau_interval, iqm as iqm_rows, verdict as plateau_verdict  # noqa: E402
from hiring.config import CONFIG_DIR, PRIMARY_SETTING  # noqa: E402
from hiring.decision import BOOT, STATS, adjust, compare, welch  # noqa: E402
from score_final import FINAL, FULL_SETTINGS, ROOT, STANDIN, best_rule, scenario, tuned_rules  # noqa: E402

ACTORS = ("index", "unrestricted", "pooled")
H0_LEVELS = ["h0_0", "h0_0.02", "h0_0.05", "h0_0.10"]
PRIMARY = f"{PRIMARY_SETTING}_survival"


def choices(stand_in: bool) -> tuple:
    spec = yaml.safe_load((CONFIG_DIR / "decision_rule.yaml").read_text())
    undecided = [k for k, v in spec.items() if v["decided"] is None]
    if undecided and not stand_in:
        sys.exit("Refused: these points of the pre-registered wording are not decided (configs/decision_rule.yaml):\n  "
                 + "\n  ".join(f"{k}: {spec[k]['options']}" for k in undecided))
    c = {k: (v["stand_in"] if v["decided"] is None else v["decided"]) for k, v in spec.items()}
    for k, v in c.items():
        assert v in spec[k]["options"], f"{k}: {v} is not one of {spec[k]['options']}"
    return c, undecided


# ---- reading scores -----------------------------------------------------------------------------------

class Scores:
    def __init__(self, root: Path):
        self.root, self.episodes = root, {}

    def _check(self, setting, d):
        """Every score at one setting must be on the same episodes in the same order (pairing)."""
        ref = self.episodes.setdefault(setting, d["episode_seed"])
        assert np.array_equal(ref, d["episode_seed"]), f"scores at {setting} are not on the same episodes"
        return d["ret"]

    def rule(self, setting, label):
        return self._check(setting, np.load(self.root / "rules" / setting / f"{label}.npz"))

    def runs(self, group) -> dict:
        """{actor: [(seed, folder), ...]} of the runs of a group, by seed."""
        out = {}
        for meta in sorted((self.root / group).glob("*/run.json")):
            actor, seed = meta.parent.name.rsplit("_s", 1)
            out.setdefault(actor, []).append((int(seed), meta.parent))
        return {a: sorted(v) for a, v in out.items()}

    def agents(self, group, setting) -> dict:
        """{actor: (seeds x episodes) greedy payoffs}."""
        return {a: np.stack([self._check(setting, np.load(f / "eval.npz")) for _, f in runs])
                for a, runs in self.runs(group).items()}


def missing_runs(root: Path) -> list:
    """Registered runs (configs/final_runs.yaml, seeds.json of each block) that are not scored yet."""
    spec = yaml.safe_load((CONFIG_DIR / "final_runs.yaml").read_text())
    out = []
    for b in spec["blocks"]:
        folder = root / f"{b['setting']}_{b['belief']}"
        if not (folder / "seeds.json").exists():
            out.append(f"{folder.name}: not started")
            continue
        out += [f"{folder.name}/{a}_s{s}" for a in spec["actors"] for s in json.loads((folder / "seeds.json").read_text())
                if not (folder / f"{a}_s{s}" / "eval.npz").exists()]
    ab = spec["ablations"]
    out += [f"ablations/{arm['name']}/{arm['actor']}_s{s}" for arm in ab["arms"] for s in ab["seeds"]
            if not (root / "ablations" / arm["name"] / f"{arm['actor']}_s{s}" / "eval.npz").exists()]
    return out


def bound(setting: str) -> float:
    sc = scenario(setting)
    for s in json.loads((ROOT / "results" / "g2" / "bounds.json").read_text())["settings"]:
        if (s["scenario"], s["h0"], s["accept_decay"]) == (sc.name, sc.h0, sc.accept_decay):
            return s["bound"]
    raise KeyError(setting)


def episodes_as_units(x):
    """A rule has no training seed: its unit is the episode. (episodes,) -> (units, 1)."""
    return np.asarray(x, float)[:, None]


def boot(draw, seed=0) -> tuple:
    rng = np.random.default_rng(seed)
    return tuple(float(x) for x in np.percentile([draw(rng) for _ in range(BOOT)], [2.5, 97.5]))


def pick(v, rng):
    return v[rng.integers(0, len(v), len(v))]


def fmt(c) -> str:
    return f"{c['diff']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}], margin {c['margin']:.4f}: {c['verdict']}"


def increases(points, ends, c) -> bool:
    """"Grows with h0": every step up (points) or last level minus first with its interval above zero."""
    return bool(np.all(np.diff(points) > 0)) if c["trend"] == "monotone_points" else ends[0] > 0


# ---- the hypotheses -------------------------------------------------------------------------------------

def h1a() -> dict:
    d = json.loads((ROOT / "results" / "g3" / "rules_small.json").read_text())
    top = max(d["rows"], key=lambda r: r["tuning_mean"])              # chosen on the tuning episodes
    return dict(verdict="holds" if top["percent_of_optimum"] >= 95 else "does not hold",
                lines=[f"best tuned rule ({top['family']}, {top['belief']}): exact value {top['exact_value']:.5f}, "
                       f"{top['percent_of_optimum']:.2f}% of the optimum {d['optimum']:.5f} (asked: at least 95%)"])


def h1b(S: Scores, c) -> dict:
    lines, share, short = [], {}, {}
    for s in FULL_SETTINGS:
        family, belief, _ = best_rule(s)
        ret, b = S.rule(s, f"{family}__{belief}"), bound(s)
        share[s] = 100 * ret.mean() / b
        short[s] = (100 * (1 - ret / b)) if c["h1b_shortfall"] == "percent" else (b - ret)       # per episode; its mean is the shortfall
        half = 1.96 * 100 * ret.std(ddof=1) / np.sqrt(len(ret)) / b
        lines.append(f"{s:11s} best rule {family} ({belief}) {ret.mean():.4f}, bound {b:.4f}: {share[s]:.2f}% (+-{half:.2f})")
    which = dict(all_full=FULL_SETTINGS, h0_levels=H0_LEVELS, primary=[PRIMARY_SETTING])[c["h1b_settings"]]
    in_range = all(85 <= share[s] <= 91 for s in which)
    points = [short[s].mean() for s in H0_LEVELS]
    ends = boot(lambda rng: pick(short[H0_LEVELS[-1]], rng).mean() - pick(short[H0_LEVELS[0]], rng).mean())
    grows = increases(points, ends, c)
    lines += [f"inside 85 to 91% at {', '.join(which)}: {'yes' if in_range else 'no'}",
              f"shortfall ({c['h1b_shortfall']}) at h0 = 0, 0.02, 0.05, 0.10: " + ", ".join(f"{p:.3f}" for p in points)
              + f"; last minus first [{ends[0]:+.3f}, {ends[1]:+.3f}]; grows with h0: {'yes' if grows else 'no'}"]
    return dict(verdict="holds" if in_range and grows else "does not hold", in_range=in_range, grows=grows, lines=lines)


def per_actor(c, results: dict, holds) -> str:
    """One verdict for a hypothesis about "PPO", from {actor: decision-rule verdict}."""
    if c["ppo_actor"] != "each":
        return holds(results[c["ppo_actor"]])
    return "; ".join(f"{a}: {holds(v)}" for a, v in results.items())


def h2a(S: Scores, c, level) -> dict:
    lines, out = [], {}
    for actor, runs in S.runs("small_survival").items():
        e = [json.loads((f / "exact.json").read_text()) for _, f in runs]
        vals, opt = np.array([x["exact_value"] for x in e]), e[0]["optimum"]
        r = compare(vals[:, None], [[opt]], resample="seeds", stat=c["statistic"], level=level)
        if c["h2a_basis"] == "mean_mark":
            out[actor] = "equivalent" if vals.mean() >= 0.98 * opt else "worse"
        else:
            out[actor] = r["verdict"]
        lines.append(f"{actor:13s} {len(vals)} seeds, exact values {vals.min():.5f} to {vals.max():.5f}, mean {100 * vals.mean() / opt:.2f}% "
                     f"of the optimum; minus optimum {fmt(r)}" + ("" if c["h2a_basis"] == "interval" else f"; mean against the 98% mark: {out[actor]}"))
    holds = lambda v: {"equivalent": "holds", "worse": "does not hold"}.get(v, "inconclusive")
    return dict(verdict=per_actor(c, out, holds), by_actor=out, lines=lines)


def h2b(S: Scores, c, level) -> dict:
    family, belief, _ = best_rule(PRIMARY_SETTING)
    rule = S.rule(PRIMARY_SETTING, f"{family}__{belief}")
    lines, out, p = [f"comparator: {family} ({belief}), {rule.mean():.4f}"], {}, {}
    for actor, m in S.agents(PRIMARY, PRIMARY_SETTING).items():
        r = compare(m, rule[None, :], resample=c["resample"], stat=c["statistic"], level=level)
        out[actor], p[f"H2b {actor}"] = r["verdict"], welch(m.mean(axis=1), [rule.mean()])
        lines.append(f"{actor:13s} {len(m)} seeds, minus rule {fmt(r)}; t-test against the rule's score p = {p[f'H2b {actor}']:.4f}")
    holds = lambda v: {"equivalent": "holds", "better": "holds", "worse": "does not hold"}.get(v, "inconclusive")
    return dict(verdict=per_actor(c, out, holds), by_actor=out, lines=lines, p=p)


def h3(S: Scores, c, level) -> dict:
    levels = [c["h3_zero_setting"]] + H0_LEVELS[1:]
    lines, res, p, means = [], {}, {}, {}
    for s in levels:
        m = S.agents(f"{s}_survival", s)
        u, i = m["unrestricted"], m["index"]
        res[s] = compare(u, i, resample=c["resample"], pair_seeds=c["pair_seeds"], stat=c["statistic"], level=level)
        means[s] = (u.mean(axis=1), i.mean(axis=1))
        p[f"H3 {s}"] = welch(*means[s])
        lines.append(f"{s:9s} unrestricted minus index {fmt(res[s])}; Welch p = {p[f'H3 {s}']:.4f}")
    f = STATS[c["statistic"]]
    (u1, i1), (u0, i0) = means[levels[-1]], means[levels[0]]
    ends = boot(lambda rng: (f(pick(u1, rng)) - f(pick(i1, rng))) - (f(pick(u0, rng)) - f(pick(i0, rng))))
    grows = increases([res[s]["diff"] for s in levels], ends, c)
    at_zero = res[levels[0]]["verdict"]
    lines.append(f"difference at {', '.join(levels)}: " + ", ".join(f"{res[s]['diff']:+.4f}" for s in levels)
                 + f"; last minus first [{ends[0]:+.4f}, {ends[1]:+.4f}]; increases with h0: {'yes' if grows else 'no'}")
    v = "holds" if at_zero == "equivalent" and grows else ("inconclusive" if at_zero == "inconclusive" and grows else "does not hold")
    return dict(verdict=v, at_zero=at_zero, grows=grows, lines=lines, p=p)


def h4a(S: Scores, c) -> dict:
    lines, ok = [], True
    for s in ("h0_0.05", "h0_0.10"):
        families = [best_rule(s)[0]] if c["h4a_families"] == "best_rule" else sorted({f for f, _ in tuned_rules(s)})
        for fam in families:
            other = f"{fam}__naive" if c["h4a_pairing"] == "tuned" else f"{fam}__naive_swapped"
            r = compare(episodes_as_units(S.rule(s, f"{fam}__survival")), episodes_as_units(S.rule(s, other)), pair_seeds=True)
            raised = r["verdict"] == "better" if c["h4a_raises"] == "better" else r["lo"] > 0
            d = np.load(S.root / "rules" / s / f"{fam}__share_{c['h4a_pairing']}.npz")
            ch, cd = d["changed"], d["candidate_days"]
            share = 100 * ch.sum() / cd.sum()
            lo, hi = boot(lambda rng: (lambda k: 100 * ch[k].sum() / cd[k].sum())(rng.integers(0, len(ch), len(ch))))
            measurable = lo > 0 if c["h4a_measurable"] == "interval_above_zero" else share >= 1.0
            ok = ok and raised and measurable
            lines.append(f"{s:8s} {fam:22s} survival-aware minus naive ({c['h4a_pairing']}) {fmt(r)}; raises payoff: {'yes' if raised else 'no'}; "
                         f"decision changed on {share:.2f}% of candidate-days [{lo:.2f}, {hi:.2f}] ({len(ch)} episodes): "
                         f"{'measurable' if measurable else 'not measurable'}")
    lines.append("unit: the episode (a rule has no training seed); bootstrap over paired episodes")
    return dict(verdict="holds" if ok else "does not hold", lines=lines)


def steps_to(folder: Path, target: float) -> float:
    """Training steps at the first curve evaluation at or above the target; if never, one evaluation past the end."""
    ev = np.load(folder / "evaluations.npz")
    t, curve = ev["timesteps"], ev["results"].mean(axis=1)
    hit = np.flatnonzero(curve >= target)
    return float(t[hit[0]] if len(hit) else t[-1] + (t[-1] - t[-2]))


def h4b(S: Scores, c, level) -> dict:
    naive, surv = S.agents(f"{PRIMARY_SETTING}_naive", PRIMARY_SETTING), S.agents(PRIMARY, PRIMARY_SETTING)
    runs_n, runs_s = S.runs(f"{PRIMARY_SETTING}_naive"), S.runs(PRIMARY)
    refs = json.loads((S.root / "rules" / PRIMARY_SETTING / "curve_refs.json").read_text())
    lines, verdicts, p = [], [], {}
    for actor in surv:
        a, b = (naive[actor], surv[actor]) if c["h4b_comparator"] == "survival" else (surv[actor], naive[actor])
        r = compare(a, b, resample=c["resample"], pair_seeds=c["pair_seeds"], stat=c["statistic"], level=level)
        p[f"H4b {actor}"] = welch(a.mean(axis=1), b.mean(axis=1))
        if c["h4b_target"] == "rule_98":
            target = 0.98 * refs["best_rule"]
        else:
            target = 0.95 * float(np.mean([np.load(f / "evaluations.npz")["results"][-1].mean() for _, f in runs_s[actor]]))
        sn = np.array([steps_to(f, target) for _, f in runs_n[actor]])
        ss = np.array([steps_to(f, target) for _, f in runs_s[actor]])
        lo, hi = boot(lambda rng: np.median(pick(sn, rng)) - np.median(pick(ss, rng)))
        faster = np.median(ss) < np.median(sn) if c["h4b_speed"] == "median_points" else lo > 0
        v = "holds" if r["verdict"] == "equivalent" and faster else ("inconclusive" if r["verdict"] == "inconclusive" and faster else "does not hold")
        verdicts.append(f"{actor}: {v}")
        lines.append(f"{actor:13s} {'naive minus survival-aware' if c['h4b_comparator'] == 'survival' else 'survival-aware minus naive'} {fmt(r)}; "
                     f"Welch p = {p[f'H4b {actor}']:.4f}; steps to {target:.3f} (median): survival-aware {np.median(ss):,.0f}, naive {np.median(sn):,.0f}, "
                     f"naive minus survival-aware [{lo:+,.0f}, {hi:+,.0f}]; fewer steps: {'yes' if faster else 'no'}")
    return dict(verdict="; ".join(verdicts), lines=lines, p=p)


def plateau(S: Scores) -> list:
    """The +-1% plateau rule per setting and actor (final model minus the model at 80% of training)."""
    rng, lines = np.random.default_rng(0), []
    for group in sorted(p.name for p in S.root.iterdir() if (p / "seeds.json").exists() or p.name.endswith(("_survival", "_naive"))):
        for actor, runs in S.runs(group).items():
            d = [np.load(f / "plateau.npz") for _, f in runs]
            diff = np.stack([x["final"] - x["at_80"] for x in d])
            final = float(iqm_rows(np.array([x["final"].mean() for x in d])))
            lo, hi = plateau_interval(diff, rng)
            lines.append(f"{group:20s} {actor:13s} {len(d):2d} runs, change over the last 20% {float(iqm_rows(diff.mean(axis=1))):+.4f} "
                         f"[{lo:+.4f}, {hi:+.4f}], band +-{0.01 * abs(final):.4f}: {plateau_verdict(lo, hi, final)}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stand-in", action="store_true")
    a = ap.parse_args()
    root = STANDIN if a.stand_in else FINAL
    c, undecided = choices(a.stand_in)
    if not a.stand_in and (gaps := missing_runs(root)):
        sys.exit(f"Refused: {len(gaps)} registered runs are not scored yet (no seed is dropped), for example:\n  " + "\n  ".join(gaps[:10]))
    S = Scores(root)
    family = 3 if c["multiplicity_family"] == "h2b" else 3 + len(H0_LEVELS) + 3
    widened = 1 - 0.05 / family if c["multiplicity_form"] == "bonferroni_intervals" else 0.95
    lvl = lambda member: widened if member else 0.95          # H3 and H4b are in the family only under "all_ppo"
    wide = c["multiplicity_family"] == "all_ppo"
    H = dict(H1a=h1a(), H1b=h1b(S, c), H2a=h2a(S, c, 0.95), H2b=h2b(S, c, lvl(True)), H3=h3(S, c, lvl(wide)),
             H4a=h4a(S, c), H4b=h4b(S, c, lvl(wide)))
    p = dict(H["H2b"]["p"], **(dict(H["H3"]["p"], **H["H4b"]["p"]) if wide else {}))
    method = "holm" if c["multiplicity_form"] == "holm_p" else "bonferroni"
    adjusted = dict(zip(p, adjust(list(p.values()), method)))

    head = ["STAND-IN: tuning-seed models on development episodes. A PIPELINE TEST, NOT RESULTS." if a.stand_in
            else "Final results: evaluation episodes, final training seeds.",
            "Decision rule: 95% bootstrap interval of the paired difference against a margin of 2% of the comparator's payoff;",
            "equivalent (inside the margin) takes precedence, then better, worse, inconclusive (PREREGISTRATION.md).", "",
            "Choices where the pre-registered wording is open (configs/decision_rule.yaml):"]
    head += [f"  {k} = {v}" + ("   <- NOT DECIDED, stand-in value" if k in undecided else "") for k, v in c.items()]
    lines = head + [""]
    for name, h in H.items():
        lines += [f"{name}: {h['verdict'].upper()}"] + [f"    {x}" for x in h["lines"]] + [""]
    lines += [f"Multiplicity ({method}, family of {len(p)}): " + "; ".join(f"{k} p = {p[k]:.4f}, adjusted {adjusted[k]:.4f}" for k in p)]
    if c["multiplicity_form"] == "bonferroni_intervals":
        lines.append(f"The intervals of the family's comparisons above are at the {100 * widened:.2f}% level.")
    lines += ["", "Plateau test (flat: the 95% interval of the change lies inside +-1% of the final payoff):"] + [f"    {x}" for x in plateau(S)]
    text = "\n".join(lines)
    print(text)
    (root / "verdicts.txt").write_text(text + "\n")
    (root / "verdicts.json").write_text(json.dumps(dict(
        stand_in=a.stand_in, choices=c, undecided=undecided, p=p, p_adjusted=adjusted,
        hypotheses={k: {x: y for x, y in h.items() if x != "p"} for k, h in H.items()}), indent=1, default=str))


if __name__ == "__main__":
    main()
