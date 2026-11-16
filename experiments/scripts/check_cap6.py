"""Interview-cap check for a trained agent without pressure (standing rule: above 5% of hires at
the cap of 3, rerun with a cap of 6).

The unrestricted actor at no pressure, 10 million steps, seed 101, has 12.5% of hires at the cap of
3. A trained agent cannot be re-scored with another cap (the cap is part of its observation), so
one agent was trained with a cap of 6. Decision rule (Nov 15): if it lands within 0.02 of the
cap-3 agent, record it and keep the cap at 3.

Each agent is scored in the environment it was trained in, on the same 4,000 development episodes
(the episode draws do not depend on the cap), greedy actions, masks on.
Writes results/g2/cap6_check.{txt,json}.
"""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hiring.config import episode_seeds, load_scenario  # noqa: E402
from hiring.evaluation import PPOPolicy, evaluate, paired_compare, summarize  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "g2"
N, LIMIT = 4000, 0.02
RUNS = {3: OUT / "ppo_no_pressure_10m" / "unrestricted_s101", 6: OUT / "ppo_no_pressure_cap6" / "unrestricted_s101"}


def job(cap):
    import torch as th
    th.set_num_threads(1)
    cfg = json.loads((RUNS[cap] / "run.json").read_text())["config"]
    sc = load_scenario(cfg["scenario"], **(cfg.get("scenario_overrides") or {}))
    assert sc.max_interviews == cap
    return cap, evaluate(sc, PPOPolicy.load(RUNS[cap] / "model.zip"), seeds=episode_seeds("dev", N))


def main():
    with Pool(2) as pool:
        res = dict(pool.map(job, [3, 6]))
    c = paired_compare(res[6]["ret"], res[3]["ret"])
    lines = ["Unrestricted actor, no pressure, 10 million steps, seed 101: interview cap 3 against cap 6.", ""]
    rows = {}
    for cap in (3, 6):
        sm = summarize(res[cap])
        rows[cap] = dict(mean=sm["mean"], se=sm["se"], interviews=sm["interviews_per_episode"], hires=sm["hires_per_episode"],
                         at_cap=sm["share_hires_at_interview_cap"])
        lines.append(f"  cap {cap}: {sm['mean']:.3f} +- {sm['se']:.3f}  interviews {sm['interviews_per_episode']:.1f}, "
                     f"hires {sm['hires_per_episode']:.2f}, hires at the cap {100 * sm['share_hires_at_interview_cap']:.1f}%")
    close = abs(c["diff"]) <= LIMIT
    lines += ["", f"cap 6 minus cap 3 (paired, {N:,} development episodes): {c['diff']:+.3f}, 95% interval [{c['lo']:+.3f}, {c['hi']:+.3f}]",
              f"Decision rule: within {LIMIT} -> " + ("YES: recorded, the cap stays at 3." if close else "NO: outside 0.02; reported to the user."),
              "", "Two separately trained agents, one seed each: the difference includes training-seed luck.",
              "Tuning seed, development episodes: a check, not a reported result."]
    text = "\n".join(lines)
    print(text)
    (OUT / "cap6_check.txt").write_text(text + "\n")
    (OUT / "cap6_check.json").write_text(json.dumps(dict(episodes=N, caps={str(k): v for k, v in rows.items()}, diff=c["diff"],
                                                        interval=[c["lo"], c["hi"]], within_limit=bool(close)), indent=1))


if __name__ == "__main__":
    main()
