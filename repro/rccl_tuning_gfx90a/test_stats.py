#!/usr/bin/env python3
"""Checks on the statistics, including the one that matters most: that a null
comparison comes back null. Run with test_designs.py before submitting."""
from __future__ import annotations

import math
import random
import sys

import stats


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}{'  -- ' + detail if detail and not condition else ''}")
    return condition


def test_basics() -> bool:
    ok = check("geomean of [1,4] is 2", abs(stats.geomean([1, 4]) - 2.0) < 1e-12)
    ok &= check("geomean ignores non-positive values",
                abs(stats.geomean([0, -1, 9, 1]) - 3.0) < 1e-12)
    ok &= check("percentile interpolates", abs(stats.percentile([0, 10], 0.5) - 5.0) < 1e-12)
    ok &= check("as_percent(log 1.05) is 5%", abs(stats.as_percent(math.log(1.05)) - 5.0) < 1e-9)
    ok &= check("t95 falls back to normal past df 30", stats.t95(200) == 1.960)
    slope, _ = stats.ols_slope([1, 2, 3, 4], [2, 4, 6, 8])
    ok &= check("ols recovers a known slope", abs(slope - 2.0) < 1e-12)
    return ok


def test_null_stays_null() -> bool:
    """The self-check the whole study rests on: comparing noise against noise must
    not produce a winner. If this fails, the pipeline manufactures findings."""
    rng = random.Random(11)
    false_positives = 0
    trials = 400
    for _ in range(trials):
        ratios = [rng.gauss(0.0, 0.02) for _ in range(3)]
        mean, half = stats.mean_ci(ratios)
        if abs(mean) - half > 0:
            false_positives += 1
    rate = false_positives / trials
    return check(f"null comparison rejects at about 5% (got {rate:.1%})", 0.01 <= rate <= 0.10)


def test_known_effect_is_found() -> bool:
    rng = random.Random(23)
    injected = math.log(1.08)
    ratios = [injected + rng.gauss(0.0, 0.015) for _ in range(6)]
    mean, half = stats.mean_ci(ratios)
    ok = check("an injected 8% effect is recovered",
               abs(stats.as_percent(mean) - 8.0) < 2.0, f"got {stats.as_percent(mean):.1f}%")
    ok &= check("and its interval excludes zero", abs(mean) - half > 0)
    return ok


def test_variance_and_mde() -> bool:
    rng = random.Random(5)
    # 5 allocations, 10 slots each: allocation spread 6%, position spread 2%.
    blocks = [[math.log(1.0) + rng.gauss(0, 0.06) + rng.gauss(0, 0.02) for _ in range(10)]
              for _ in range(5)]
    # Re-generate with a single per-block offset so the components are separable.
    blocks = []
    for _ in range(5):
        offset = rng.gauss(0, 0.06)
        blocks.append([offset + rng.gauss(0, 0.02) for _ in range(10)])

    comp = stats.variance_components(blocks)
    ok = check("allocation CV recovered near 6%", 3.0 < comp["cv_alloc_pct"] < 10.0,
               f"got {comp['cv_alloc_pct']:.1f}%")
    ok &= check("position CV recovered near 2%", 1.0 < comp["cv_pos_pct"] < 3.5,
                f"got {comp['cv_pos_pct']:.1f}%")
    ok &= check("allocation spread exceeds position spread",
                comp["cv_alloc_pct"] > comp["cv_pos_pct"])
    ok &= check("MDE shrinks as allocations are added",
                stats.mde_percent(2.0, 3) > stats.mde_percent(2.0, 12))
    ok &= check("MDE at CV_pos=2%, k=3 is about 4.6%",
                abs(stats.mde_percent(2.0, 3) - 4.6) < 0.2,
                f"got {stats.mde_percent(2.0, 3):.2f}%")
    return ok


def test_screening_helpers() -> bool:
    # Eleven null effects and one real one: PSE must not be inflated by the real one.
    effects = [0.001, -0.002, 0.0015, -0.001, 0.002, -0.0018, 0.0012, -0.0009,
               0.0011, -0.0014, 0.0016, 0.25]
    pse = stats.lenth_pse(effects)
    ok = check("Lenth PSE resists one large real effect", pse is not None and pse < 0.01,
               f"got {pse}")

    keep = stats.benjamini_hochberg([0.001, 0.2, 0.9, 0.04], q=0.10)
    ok &= check("BH keeps the smallest p-value", keep[0])
    ok &= check("BH drops the obviously null ones", not keep[1] and not keep[2])
    ok &= check("BH preserves input order", len(keep) == 4)
    ok &= check("BH on all-null input keeps nothing",
                not any(stats.benjamini_hochberg([0.4, 0.5, 0.6, 0.99], q=0.10)))
    return ok


def main() -> int:
    suites = [("basics", test_basics),
              ("a null comparison stays null", test_null_stays_null),
              ("a known effect is recovered", test_known_effect_is_found),
              ("variance components and MDE", test_variance_and_mde),
              ("screening helpers", test_screening_helpers)]
    ok = True
    for label, fn in suites:
        print(f"\n{label}")
        ok &= fn()
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
