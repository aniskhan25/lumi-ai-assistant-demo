"""Backport upstream vLLM's MiMo fused-FP8 qkv sharding into vLLM 0.26.0.

vLLM 0.26.0 (LAIF 20260929) assumes the fused qkv_proj is stored as one
[Q|K|V] group per KV head. MiMo-V2.5/V2.6 checkpoints store it pre-sharded
into `num_key_value_heads` (= 4) chunks instead, with per-chunk FP8 block
scales. So the 0.26 loader fails at every TP size:
  TP=8  AssertionError: TP size must evenly split the number of KV heads
  TP=4  RuntimeError: size of tensor a (1856) must match ... b (1792)
Upstream main fixed this by rewriting `_shard_fp8_qkv_proj`; this script copies
that function (and its `_requantize_fp8` helper) from an upstream mimo_v2.py into
the container's file and passes `ckpt_tp` from the caller. Nothing else changes.

    python3 backport_mimo_v2.py <container mimo_v2.py> <upstream mimo_v2.py> <out>
"""

import re
import sys


def top_level_def(src, name):
    m = re.search(rf"^def {name}\(.*?(?=^(?:@|class |def )\S)", src, re.S | re.M)
    if not m:
        sys.exit(f"{name} not found")
    return m


def main():
    local, upstream, out = sys.argv[1:]
    src = open(local).read()
    up = open(upstream).read()

    new_fns = top_level_def(up, "_requantize_fp8").group(0) + top_level_def(
        up, "_shard_fp8_qkv_proj"
    ).group(0)
    old = top_level_def(src, "_shard_fp8_qkv_proj")
    src = src[: old.start()] + new_fns + src[old.end():]

    if "from vllm.utils.math_utils import cdiv" not in src:
        src = src.replace(
            "\nimport torch\n", "\nimport torch\n\nfrom vllm.utils.math_utils import cdiv\n", 1
        )

    caller = "            tp_size=tp_size,\n        )\n        sharded = {"
    if src.count(caller) != 1:
        sys.exit("caller of _shard_fp8_qkv_proj not found exactly once")
    src = src.replace(
        caller,
        "            tp_size=tp_size,\n"
        "            # The fused qkv_proj is pre-sharded for this many ranks.\n"
        "            ckpt_tp=self.config.num_key_value_heads,\n"
        "        )\n        sharded = {",
    )
    open(out, "w").write(src)


if __name__ == "__main__":
    main()
