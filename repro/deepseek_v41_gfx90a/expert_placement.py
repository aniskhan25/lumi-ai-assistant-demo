"""Per-layer router expert counts (EXPERT_COUNTS=1, hooks.py:_expert_counts) -> how unbalanced
the linear expert placement is, and a balanced placement.

Under expert parallelism each rank owns 48 of the 384 experts; each MoE layer ends in an
all-reduce that waits for the rank whose experts got the most routed tokens. The linear
placement gives rank r experts [48 r, 48 r + 48). The balanced one assigns, per layer, the
experts by count, hottest first, each to the least-loaded rank that still has room (48
experts a rank).

Reported per placement: the busiest rank's load over the mean, averaged over layers (what
each all-reduce waits for), and its sum over layers (the summed per-layer wait).

    python expert_placement.py expert_counts.json [placement_out.json]
"""

import json
import sys

EP, PER = 8, 48


def loads(c, place):
    lo = [0] * EP
    for e, r in enumerate(place):
        lo[r] += c[e]
    return lo


def balanced(c):
    order = sorted(range(len(c)), key=lambda e: -c[e])
    lo, cnt, place = [0] * EP, [0] * EP, [0] * len(c)
    for e in order:
        r = min((r for r in range(EP) if cnt[r] < PER), key=lambda r: lo[r])
        place[e], lo[r], cnt[r] = r, lo[r] + c[e], cnt[r] + 1
    return place


def layer_key(name):
    digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
    return (name.split(".layers.")[0], int(digits[0]) if digits else -1)


def main() -> None:
    counts = json.load(open(sys.argv[1]))
    lin_ratio, bal_ratio, lin_excess, bal_excess = [], [], 0.0, 0.0
    per_rank_lin = [0.0] * EP
    total = 0
    placements = {}
    print(f"{len(counts)} MoE layers")
    import os

    target = int(os.environ.get("TARGET_LAYERS", 40))  # 40-42: the DSpark drafter's MoE layers
    for name in sorted(counts, key=layer_key):
        if layer_key(name)[1] >= target:
            continue
        c = counts[name]
        n = sum(c)
        if n == 0:
            continue
        total += n
        lin = loads(c, [e // PER for e in range(len(c))])
        place = balanced(c)
        bal = loads(c, place)
        mean = n / EP
        lin_ratio.append(max(lin) / mean)
        bal_ratio.append(max(bal) / mean)
        lin_excess += max(lin) - mean
        bal_excess += max(bal) - mean
        for r in range(EP):
            per_rank_lin[r] += lin[r] / mean
        placements[name] = place
        top = sorted(c, reverse=True)
        print(f"  {name:<40} routed {n:>10}  hottest expert {top[0] / (n / len(c)):4.1f}x mean, "
              f"top-10 share {100 * sum(top[:10]) / n:4.1f} %;  busiest rank / mean: linear "
              f"{lin_ratio[-1]:.3f}, balanced {bal_ratio[-1]:.3f}")

    L = len(lin_ratio)
    print(f"\nrouted assignments counted: {total:,}")
    print("busiest rank / mean, averaged over layers: linear "
          f"{sum(lin_ratio) / L:.3f}, balanced {sum(bal_ratio) / L:.3f}")
    print("each rank's load / mean, averaged over layers (linear): "
          + " ".join(f"r{r} {per_rank_lin[r] / L:.3f}" for r in range(EP)))
    print(f"summed per-layer excess of the busiest rank over the mean, in routed assignments: "
          f"linear {lin_excess:,.0f}, balanced {bal_excess:,.0f} "
          f"({100 * (1 - bal_excess / lin_excess):.0f} % less waiting, if MoE time is "
          "proportional to routed tokens)")
    if len(sys.argv) > 2:
        json.dump(placements, open(sys.argv[2], "w"))
        print(f"placement written to {sys.argv[2]}")


if __name__ == "__main__":
    main()
