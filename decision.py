"""
Turning model scores into pass / review / fail decisions and euros.

Shared by train.py (to sweep and pick thresholds) and app.py (to recompute
everything live when the sliders move), so both always agree.

Base-rate correction
--------------------
The splits do not contain cracked eggs in the proportion a real station sees
them. So we never cost the raw confusion matrix directly. We measure, from the
split, how the model treats a cracked egg and how it treats a sound egg
(rates conditional on the true class), then reweight them by the base rate we
expect in production. That is what "expected cost per 1,000 eggs" means here.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

PASS, REVIEW, FAIL = 0, 1, 2
OUTCOMES = ["pass", "review", "fail"]


@dataclass
class Costs:
    fn: float = config.COST_FN
    fp: float = config.COST_FP
    tp: float = config.COST_TP
    tn: float = config.COST_TN
    review: float = config.COST_REVIEW
    human_recall: float = config.HUMAN_RECALL
    human_false_fail: float = config.HUMAN_FALSE_FAIL


def decide(scores, t_low: float, t_high: float) -> np.ndarray:
    """0 = pass, 1 = review, 2 = fail. With t_low == t_high there is no review band."""
    scores = np.asarray(scores, dtype=float)
    t_low, t_high = min(t_low, t_high), max(t_low, t_high)
    out = np.full(scores.shape, REVIEW, dtype=int)
    out[scores < t_low] = PASS
    out[scores >= t_high] = FAIL
    return out


def outcome_counts(scores, labels, t_low: float, t_high: float) -> pd.DataFrame:
    """Raw counts on the split: rows = true class, columns = pass/review/fail."""
    labels = np.asarray(labels, dtype=int)
    d = decide(scores, t_low, t_high)
    table = {
        name: [int(np.sum((d == k) & (labels == 1))), int(np.sum((d == k) & (labels == 0)))]
        for k, name in enumerate(OUTCOMES)
    }
    return pd.DataFrame(table, index=["cracked (true)", "sound (true)"])


def conditional_rates(scores, labels, t_low: float, t_high: float) -> dict:
    """P(outcome | cracked) and P(outcome | sound), estimated from the split."""
    counts = outcome_counts(scores, labels, t_low, t_high).to_numpy(dtype=float)
    pos = counts[0] / max(counts[0].sum(), 1)
    neg = counts[1] / max(counts[1].sum(), 1)
    return {
        "pos_pass": pos[PASS], "pos_review": pos[REVIEW], "pos_fail": pos[FAIL],
        "neg_pass": neg[PASS], "neg_review": neg[REVIEW], "neg_fail": neg[FAIL],
    }


def cost_per_egg(rates: dict, costs: Costs, base_rate: float) -> float:
    """Expected euros per egg at a given production base rate.

    Cracked egg:  passed -> FN, failed -> TP, reviewed -> review time, and the
                  person still misses (1 - human_recall) of them (-> FN).
    Sound egg:    passed -> TN, failed -> FP, reviewed -> review time, and the
                  person wrongly rejects human_false_fail of them (-> FP).
    """
    c = costs
    cracked = (
        rates["pos_pass"] * c.fn
        + rates["pos_fail"] * c.tp
        + rates["pos_review"] * (c.review + c.human_recall * c.tp + (1 - c.human_recall) * c.fn)
    )
    sound = (
        rates["neg_pass"] * c.tn
        + rates["neg_fail"] * c.fp
        + rates["neg_review"] * (c.review + c.human_false_fail * c.fp + (1 - c.human_false_fail) * c.tn)
    )
    return base_rate * cracked + (1 - base_rate) * sound


def review_share(rates: dict, base_rate: float) -> float:
    return base_rate * rates["pos_review"] + (1 - base_rate) * rates["neg_review"]


def expected_matrix_per_1000(rates: dict, base_rate: float) -> pd.DataFrame:
    """What 1,000 production eggs would look like: rows = truth, columns = outcome."""
    pos, neg = 1000 * base_rate, 1000 * (1 - base_rate)
    return pd.DataFrame(
        {
            "pass": [pos * rates["pos_pass"], neg * rates["neg_pass"]],
            "review": [pos * rates["pos_review"], neg * rates["neg_review"]],
            "fail": [pos * rates["pos_fail"], neg * rates["neg_fail"]],
        },
        index=["cracked (true)", "sound (true)"],
    ).round(1)


def evaluate(scores, labels, t_low, t_high, costs: Costs, base_rate: float) -> dict:
    rates = conditional_rates(scores, labels, t_low, t_high)
    return {
        "t_low": float(min(t_low, t_high)),
        "t_high": float(max(t_low, t_high)),
        "cost_per_1000": 1000 * cost_per_egg(rates, costs, base_rate),
        "review_share": review_share(rates, base_rate),
        **rates,
    }


GRID = np.round(np.arange(0.0, 1.0001, 0.01), 2)


def sweep_single(scores, labels, costs: Costs, base_rate: float, grid=GRID) -> pd.DataFrame:
    """One threshold, no review band: the model decides alone."""
    rows = [evaluate(scores, labels, t, t, costs, base_rate) for t in grid]
    df = pd.DataFrame(rows).rename(columns={"t_low": "threshold"}).drop(columns="t_high")
    df["recall"] = df["pos_fail"]
    df["false_fail_rate"] = df["neg_fail"]
    return df


def best_single(sweep: pd.DataFrame) -> pd.Series:
    """Cheapest threshold. Ties are broken towards the middle of the flat region,
    so a lucky edge of a small split is not mistaken for a precise optimum."""
    best = sweep["cost_per_1000"].min()
    tied = sweep[np.isclose(sweep["cost_per_1000"], best)]
    return tied.iloc[len(tied) // 2]


def sweep_band(scores, labels, costs: Costs, base_rate: float, grid=GRID) -> pd.DataFrame:
    """Every (t_low, t_high) pair with t_low <= t_high."""
    rows = []
    for i, lo in enumerate(grid):
        for hi in grid[i:]:
            rows.append(evaluate(scores, labels, lo, hi, costs, base_rate))
    return pd.DataFrame(rows)


def best_band(band: pd.DataFrame) -> pd.Series:
    """Cheapest band; among equally cheap bands, the one sending fewest to a person."""
    best = band["cost_per_1000"].min()
    tied = band[np.isclose(band["cost_per_1000"], best)]
    return tied.sort_values(["review_share", "t_low"]).iloc[0]


# --- Staffing and the alternatives -------------------------------------------

def staffing(share: float, eggs_per_day: int = config.EGGS_PER_DAY,
             seconds: float = config.SECONDS_PER_REVIEW,
             hours: float = config.PRODUCTIVE_HOURS_PER_SHIFT) -> dict:
    per_day = share * eggs_per_day
    person_hours = per_day * seconds / 3600
    return {
        "eggs_reviewed_per_day": per_day,
        "person_hours_per_day": person_hours,
        "people_needed": person_hours / hours,
    }


def alternatives(model_alone: dict, model_review: dict, costs: Costs, base_rate: float) -> pd.DataFrame:
    """Price the chosen design against the three alternatives the brief names."""
    nothing = {"pos_pass": 1, "pos_review": 0, "pos_fail": 0, "neg_pass": 1, "neg_review": 0, "neg_fail": 0}
    everyone = {"pos_pass": 0, "pos_review": 1, "pos_fail": 0, "neg_pass": 0, "neg_review": 1, "neg_fail": 0}
    options = {
        "Do nothing (pack every egg)": nothing,
        "A person checks every egg": everyone,
        "Model decides alone": model_alone,
        "Model + human review band": model_review,
    }
    rows = []
    for name, r in options.items():
        per_1000 = 1000 * cost_per_egg(r, costs, base_rate)
        share = review_share(r, base_rate)
        staff = staffing(share)
        rows.append({
            "option": name,
            "cost_per_1000_eggs": round(per_1000, 2),
            "cost_per_day_eur": round(per_1000 * config.EGGS_PER_DAY / 1000, 2),
            "share_to_person": round(share, 4),
            "people_needed": round(staff["people_needed"], 2),
            "cracked_eggs_shipped_per_1000": round(
                1000 * base_rate * (r["pos_pass"] + r["pos_review"] * (1 - costs.human_recall)), 2),
        })
    return pd.DataFrame(rows)
