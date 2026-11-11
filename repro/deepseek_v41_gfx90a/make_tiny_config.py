"""Write a tiny random-weight DeepSeek-V4.1 model directory for bring-up (--load-format dummy).

    python make_tiny_config.py <V4.1 snapshot dir> <out dir>

Keeps the real attention geometry (MLA head 512 / rope 64, 64 heads, the 32-head
indexer, 4-stream hyper-connections), quantization (MXFP8 dense, MXFP4 experts),
vocabulary and tokenizer (Engram's compressed token map is built from it), and shrinks
everything else. Eight backbone layers cover every V4.1 layer kind:

  layer        0    1       2          3, 4        5                   6       7
  ratio        0    0       2          2           1                   1       1
  role         SWA  SWA+    C2A KV +   C2A, reuse  C1A KV + indexer +  reuse   indexer
                    Engram  indexer    (4: Engram) candidate source            (masked)
"""

import json
import pathlib
import shutil
import sys

LAYERS = 8


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def engram_num_embeddings(cfg: dict) -> list[int]:
    """Sum of the per-(n-gram, head) prime bucket sizes, as EngramLayout draws them."""
    seen: set[int] = set()
    sums = []
    for _ in cfg["engram_layer_ids"]:
        total = 0
        for _ in range(cfg["engram_max_ngram_size"] - 1):
            current = cfg["engram_vocab_size"] - 1
            for _ in range(cfg["engram_n_heads"]):
                current += 1
                while not _is_prime(current) or current in seen:
                    current += 1
                seen.add(current)
                total += current
        sums.append(total)
    return sums


def main() -> None:
    src, out = (pathlib.Path(a) for a in sys.argv[1:3])
    full = json.loads((src / "config.json").read_text())
    text = dict(full["text_config"])
    text.update(
        hidden_size=1024,
        moe_intermediate_size=512,
        num_hidden_layers=LAYERS,
        n_routed_experts=16,
        compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1] + [0] * text["num_nextn_predict_layers"],
        kv_source_layer_ids=[2, 5],
        index_source_layer_ids=[2, 5, 7],
        candidate_source_layer_id=5,
        engram_layer_ids=[1, 4],
        engram_vocab_size=1000,
        dspark_target_layer_ids=[5, 6, 7],
    )
    text["engram_num_embeddings"] = engram_num_embeddings(text)
    cfg = {k: v for k, v in full.items() if k != "vision_config"}
    cfg["text_config"] = text
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json"):
        if (src / name).exists():
            shutil.copyfile(src / name, out / name)
    print(f"wrote {out}: engram_num_embeddings={text['engram_num_embeddings']}")


if __name__ == "__main__":
    main()
