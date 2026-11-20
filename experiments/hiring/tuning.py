"""Rule tuning on the tuning episodes.

ONE CONFIGURATION (PREREGISTRATION.md, decided Nov 14) is one full parameter vector of a rule
family, scored on all 4,000 tuning episodes. A search gets a fixed number of configurations: 40 by
default, the same as a PPO architecture; 360 in the robustness search at the primary setting.

THE SEARCH is a seeded (1+1) evolution strategy, the same for every rule family:
  configuration 1   the start vector: the earlier tuned vector where one exists
                    (results/g1/v3_fullprec.json), the theory values (1, 0) for the price-directed rule
  later ones        best so far + step * scale * noise, on a random subset of the parameters;
                    the step grows by 1.3 after an improvement and shrinks by 0.85 otherwise
Every configuration and its score is kept; the tuning table is the full list.

CONTAINED RULES (decided Nov 20). Each threshold family contains the narrower ones (CHAIN), so its
best score cannot be lower, but the local search does not always reach them. After its search,
every threshold family is therefore also scored at the best vector of each family it contains,
written as its own vector (`embed`), and keeps the best. These are up to 3 evaluations on top of
the budget; the price-directed rule gets none.

Only the tuning pool can be used here. Scores of the chosen configuration on these episodes are
biased upward (it was picked on them); held-out scores come from other episodes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Scenario, episode_seeds
from .evaluation import evaluate, summarize
from .rules import INF, RULE_FAMILIES

ROOT = Path(__file__).resolve().parents[2]
BELIEFS = ("naive", "survival")

# Perturbation scale per parameter (bars 0.15, falls 0.3, role term 0.05), in the order of rules.py.
SCALES = {
    "static_threshold": [0.15],
    "resume_falling_bar": [0.15, 0.3, 0.05],
    "one_interview": [0.15, 0.15, 0.3, 0.3, 0.15, 0.3, 0.05],
    "up_to_two_interviews": [0.15, 0.15, 0.15, 0.3, 0.3, 0.15, 0.15, 0.3, 0.05],
    "price_directed": [0.1, 0.01],
}
BOUNDS = {"price_directed": ([0.3, 0.0], [2.0, 0.2])}      # price scale > 0, interview price >= 0
INTERVIEW_PRICE_STEP = 0.005                                  # the price tables are cached on this grid


# Narrowest first; each step writes a family's vector as the next family's vector (same policy).
CHAIN = ["static_threshold", "resume_falling_bar", "one_interview", "up_to_two_interviews"]
_WIDEN = [lambda v: [v[0], 0.0, 0.0],                                                # no fall, no role term
          lambda v: [v[0], v[0], v[1], v[1], INF, 0.0, v[2]],                        # never interviews
          lambda v: [v[0], v[1], v[1], v[2], v[3], v[4], INF, v[5], v[6]]]           # no second interview


def embed(narrow: str, wide: str, params) -> list:
    """The vector of `wide` that plays exactly as `narrow` does with `params`."""
    a, b = CHAIN.index(narrow), CHAIN.index(wide)
    assert a < b
    v = [float(x) for x in params]
    for step in _WIDEN[a:b]:
        v = step(v)
    return v


def earlier_key(sc: Scenario) -> str:
    """The earlier simulator's pressure level closest to this scenario."""
    if sc.name == "calibrated":
        return "h0=0.02,b=0.03"
    h0 = min((0.0, 0.02, 0.05, 0.1), key=lambda x: abs(x - sc.h0))
    return f"h0={h0},b={0.0 if (h0 == 0.0 and sc.accept_decay == 0.0) else 0.03}"


def start_vector(family: str, sc: Scenario) -> list:
    if family == "price_directed":
        return [1.0, 0.0]
    earlier = json.loads((ROOT / "results" / "g1" / "v3_fullprec.json").read_text())
    return [float(x) for x in earlier[earlier_key(sc)][family]["par_full"]]


def clean(family: str, x) -> list:
    x = np.asarray(x, float)
    if family in BOUNDS:
        x = np.clip(x, *BOUNDS[family])
    if family == "price_directed":
        x[1] = round(x[1] / INTERVIEW_PRICE_STEP) * INTERVIEW_PRICE_STEP
    return [float(v) for v in x]


def score(sc: Scenario, family: str, belief: str, params, pool: str = "tuning", episodes: int | None = None) -> dict:
    """One configuration: the rule with these parameters on the tuning episodes."""
    assert pool == "tuning", "rules are tuned on the tuning pool only"
    res = evaluate(sc, RULE_FAMILIES[family](params, belief=belief), seeds=episode_seeds(pool, episodes))
    return summarize(res)


def tune(sc: Scenario, family: str, belief: str, budget: int = 40, search_seed: int = 0,
         pool: str = "tuning", episodes: int | None = None) -> dict:
    """Run one search. Returns every configuration, in order, and the best one."""
    assert pool == "tuning", "rules are tuned on the tuning pool only"
    assert family in SCALES and belief in BELIEFS and budget >= 1
    rng = np.random.default_rng([search_seed, sorted(SCALES).index(family), BELIEFS.index(belief)])
    scale = np.asarray(SCALES[family])
    best = clean(family, start_vector(family, sc))
    best_summary = score(sc, family, belief, best, pool, episodes)
    history = [dict(configuration=1, params=best, mean=best_summary["mean"], se=best_summary["se"], accepted=True)]
    step, seen = 1.0, {tuple(best)}
    while len(history) < budget:
        for _ in range(50):                                   # skip exact repeats (rounded parameters)
            pick = rng.random(len(scale)) < 0.5
            if not pick.any():
                pick[rng.integers(len(scale))] = True
            cand = clean(family, np.asarray(best) + step * scale * rng.standard_normal(len(scale)) * pick)
            if tuple(cand) not in seen:
                break
        seen.add(tuple(cand))
        s = score(sc, family, belief, cand, pool, episodes)
        accepted = s["mean"] > best_summary["mean"]
        history.append(dict(configuration=len(history) + 1, params=cand, mean=s["mean"], se=s["se"], accepted=accepted))
        if accepted:
            best, best_summary, step = cand, s, step * 1.3
        else:
            step = max(step * 0.85, 0.05)
    return dict(scenario=sc.name, h0=sc.h0, accept_decay=sc.accept_decay, family=family, belief=belief,
                budget=budget, search_seed=search_seed, pool=pool,
                episodes=best_summary["episodes"], start=history[0], best_params=best, best=best_summary,
                best_configuration=max(h["configuration"] for h in history if h["accepted"]), history=history)


def seed_contained(sc: Scenario, result: dict, contained: dict, pool: str = "tuning", episodes: int | None = None) -> dict:
    """Score the search's family at the best vector of each family it contains; keep the best.

    `result` is a finished search (from `tune`); `contained` maps a narrower family to the best
    vector of its own search. The extra configurations are appended to the history, marked with
    `contained`. The search's own best is kept under `search_best*`. Does nothing the second time.
    """
    if "search_best" in result:
        return result
    result.update(search_best=result["best"], search_best_params=result["best_params"],
                  search_best_configuration=result["best_configuration"])
    for narrow in CHAIN[:CHAIN.index(result["family"])]:
        cand = embed(narrow, result["family"], contained[narrow])
        s = score(sc, result["family"], result["belief"], cand, pool, episodes)
        accepted = s["mean"] > result["best"]["mean"]
        result["history"].append(dict(configuration=len(result["history"]) + 1, params=cand, mean=s["mean"], se=s["se"],
                                      accepted=accepted, contained=narrow))
        if accepted:
            result.update(best=s, best_params=cand, best_configuration=len(result["history"]))
    return result
