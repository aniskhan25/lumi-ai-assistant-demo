"""Copy the upstream vLLM files the V4.1 plugin needs into vendor/vllm/, byte for byte.

    python vendor_sync.py <upstream vllm checkout> <commit>

Three groups, from the static import closure against LAIF's vLLM 0.26 (README, phase 2):
- MODEL: vllm/models/deepseek_v41 without nvidia/ and the opt-in AMD monokernel;
- NEW: model-side modules 0.26 does not have;
- SHADOW: model-side modules 0.26 has in an older version; lumi_dsv41 loads these
  instead of 0.26's. Engine-side modules are never vendored: 0.26 keeps them, and
  shims.py adds the few names upstream imports from them.
"""

import pathlib
import shutil
import sys

NEW = [
    "model_executor/kernels/attention/dsa/candidate_blocks.py",
    "model_executor/kernels/attention/dsa/fused_mqa_topk.py",
    "model_executor/kernels/attention/dsa/sparse_mqa_logits.py",
    "model_executor/layers/indexer_topk.py",
    "model_executor/layers/sparse_mqa_indexer.py",
    "models/common/ops/__init__.py",
    "models/common/ops/fused_qk_rmsnorm.py",
    "models/common/ops/sequence_parallel.py",
    "models/deepseek_v4/common/grouped_wo_a.py",
    "models/deepseek_v4/common/spec_decode.py",
    "v1/attention/backends/mla/sparse_indexer.py",
    "v1/attention/ops/metadata.py",
    # import-time helpers of the vendored Triton kernels; inert under 0.26's runner
    "model_executor/warmup/jit_warmup.py",
    "model_executor/warmup/jit_warmup_triton_helper.py",
    "transformers_utils/deepseek_v4_utils.py",
    "transformers_utils/configs/deepseek_v41.py",
    "models/common/__init__.py",
    "v1/kv_cache_layout.py",
    "config/engram.py",
]
# Engine modules: never loaded from here; shims.py execs single definitions from them
# (see GRAFTS there) inside 0.26's module of the same name.
GRAFT = [
    "config/multimodal.py",
    "config/utils.py",
    "distributed/device_communicators/shm_broadcast.py",
    "distributed/kv_transfer/kv_connector/utils.py",
    "forward_context.py",
    "model_executor/layers/fusion/quant_activation.py",
    "model_executor/layers/quantization/utils/fp8_utils.py",
    "model_executor/model_loader/weight_utils.py",
    "platforms/rocm.py",
    "utils/cpu_resource_utils.py",
    "utils/deep_gemm.py",
    "v1/attention/backend.py",
    "v1/kv_cache_interface.py",
]
SHADOW = [
    "model_executor/kernels/mhc/torch.py",
    "model_executor/kernels/mhc/triton.py",
    "model_executor/layers/mhc.py",
    "model_executor/layers/sparse_attn_indexer.py",
    "models/deepseek_v4/attention.py",
    "models/deepseek_v4/common/ops/cache_utils.py",
    "models/deepseek_v4/common/ops/fused_compress_quant_cache.py",
    "models/deepseek_v4/common/ops/fused_indexer_q.py",
    "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py",
    "models/deepseek_v4/common/ops/save_partial_states.py",
    "models/deepseek_v4/compressor.py",
    "v1/attention/backends/mla/compressor_utils.py",
    "v1/attention/backends/mla/indexer.py",
    "v1/attention/backends/mla/sparse_swa.py",
    "v1/attention/backends/mla/sparse_utils.py",
    "v1/attention/ops/common.py",
    "v1/attention/ops/rocm_aiter_mla_sparse.py",
]


def model_files(src: pathlib.Path) -> list[str]:
    root = src / "models" / "deepseek_v41"
    out = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if rel.parts[0] == "nvidia" or rel.parts[:2] == ("amd", "mono") or rel.name == "mono_decode.py":
            continue
        out.append(str(p.relative_to(src)))
    return out


def main() -> None:
    src = pathlib.Path(sys.argv[1]) / "vllm"
    commit = sys.argv[2]
    here = pathlib.Path(__file__).resolve().parent
    dst = here / "vendor" / "vllm"
    graft = here / "vendor" / "graft" / "vllm"
    for d in (dst, graft):
        if d.exists():
            shutil.rmtree(d)
    groups = {"MODEL": model_files(src), "NEW": NEW, "SHADOW": SHADOW, "GRAFT": GRAFT}
    lines = [f"vllm-project/vllm@{commit}", ""]
    for name, files in groups.items():
        lines.append(f"[{name}]")
        for rel in files:
            target = (graft if name == "GRAFT" else dst) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, target)
            lines.append(f"vllm/{rel}")
        lines.append("")
    (here / "vendor" / "UPSTREAM_V41").write_text("\n".join(lines))
    print(f"{sum(len(f) for f in groups.values())} files -> {dst}")


if __name__ == "__main__":
    main()
