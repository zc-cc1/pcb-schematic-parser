"""Develop a geometry decoder on 13 pages, then inspect the other 13 pages."""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pcb_parser.wire_segments import DEFAULTS, decode_wire_segments, export_candidate_edges, component_bridge_guard
from audit_unet_competition_geometry import match_segments

OUT = ROOT/"runs/wire_segments_v1_guarded_20261015"
EVAL = ROOT/"runs/training/unet_v7_finetune_comparison_20261015/evaluation"
AUDIT = ROOT/"runs/unet_competition_requirements_20261015/geometry_audit.json"
PRESETS = {
    "fitted": DEFAULTS | dict(maximum_gap_px=0.),
    "fitted_gap4": DEFAULTS | dict(maximum_gap_px=4.),
    "fitted_gap8": DEFAULTS | dict(maximum_gap_px=8.),
}


def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def total(rows):
    tp, fp, fn = (sum(row["metrics"][key] for row in rows) for key in ("tp", "fp", "fn"))
    return dict(pages=len(rows), tp=tp, fp=fp, fn=fn, predictions=tp+fp, references=tp+fn,
                precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn), f1=2*tp/max(1, 2*tp+fp+fn),
                page_mean_f1=float(np.mean([r["metrics"]["f1"] for r in rows])) if rows else 0,
                merged_gaps=sum(r.get("merged_gaps", 0) for r in rows))


def unmatched_analysis(points, refs, metrics):
    matched_ref = {r["reference"] for r in metrics["matches"]}
    unmatched = [r for i, r in enumerate(refs) if i not in matched_ref]
    reference_causes = dict(almost_no_mask_response=sum(r["almost_absent_5px"] for r in unmatched),
        strong_mask_evidence_but_no_segment=sum(r["supported_90pct_and_ends_5px"] for r in unmatched))
    reference_causes["partial_or_missing_end_evidence"] = len(unmatched)-sum(reference_causes.values())
    matched_pred = {r["prediction"] for r in metrics["matches"]}
    predictions = [p for i, p in enumerate(points) if i not in matched_pred]
    fragments = 0
    if predictions and refs:
        p = np.asarray(predictions); r = np.asarray([s["points"] for s in refs])
        v = r[:, 1]-r[:, 0]; squared = np.sum(v*v, axis=-1)
        delta = p[:, None, :, :]-r[None, :, None, 0, :]
        t = np.clip(np.sum(delta*v[None, :, None, :], axis=-1)/squared[None, :, None], 0, 1)
        distances = np.linalg.norm(delta-t[..., None]*v[None, :, None, :], axis=-1).max(axis=-1)
        pl = np.linalg.norm(p[:, 1]-p[:, 0], axis=-1)
        rl = np.sqrt(squared)
        fragments = int(np.any((distances <= 5) & (pl[:, None] < rl[None, :]-5), axis=1).sum())
    return dict(unmatched_reference=reference_causes, unmatched_predictions=len(predictions),
                possible_truncated_or_excess_fragment_predictions=fragments,
                scope="geometric hints against unverified source segmentation; not certified error labels")


def decode_case(variant, case, cfg, reference):
    folder = EVAL/variant/case
    mask = np.asarray(Image.open(folder/"wire_mask.png")) > 0
    pins = [p["point"] for p in read(folder/"pin_candidates.json") if p["confidence"] >= .8]
    components = read(ROOT/f"runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json")
    image_file = ROOT/f"data/standardized/{case}/{case}.png"
    assert components["image_sha256"] == sha(image_file) and components["coordinate_origin"] == "top_left"
    assert components["size"] == list(mask.shape[::-1])
    guard = component_bridge_guard(mask.shape, components["components"])
    # Decoder sees only mask/predicted tips. Reference points enter the scorer below.
    result = decode_wire_segments(mask.astype(np.float32), cfg, pin_points=pins, bridge_blocked_mask=guard)
    points = [r["points"] for r in result["segments"]]
    metrics = match_segments(points, reference)
    return result, dict(case_id=case, metrics=metrics, merged_gaps=result["summary"]["merged_gaps"],
                        error_hints=unmatched_analysis(points, reference, metrics))


def render(case, old, new, reference):
    image = np.asarray(Image.open(ROOT/f"data/standardized/{case}/{case}.png").convert("RGB"))
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), dpi=130, layout="constrained")
    for ax, title, points in zip(axes, ("原图与源参考", "原始骨架解码", "新几何解码"),
                                  ([s["points"] for s in reference], old, new)):
        ax.imshow(image, alpha=.48)
        for a, b in np.asarray(points):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#dd6b16" if title=="原图与源参考" else "#00965d", lw=.9)
        ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0); ax.axis("off"); ax.set_title(title)
    fig.suptitle(f"{case} · 源JSON仅用于左图与离线评分，解码器不读取参考")
    fig.savefig(OUT/f"comparison_{case}.png", facecolor="white"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = read(AUDIT); split = read(ROOT/"runs/unet_inference_v2/split.json")
    groups = dict(development=split["development"], check13=split["acceptance_holdout"])
    all_cases = groups["development"]+groups["check13"]
    by_variant = {v: {r["case_id"]: r for r in rows} for v, rows in audit["variants"].items()}
    protected = audit["protected_sha256"] | {str(AUDIT): sha(AUDIT)}
    for variant in by_variant:
        for case in all_cases:
            file = EVAL/variant/case/"pin_candidates.json"; protected[str(file)] = sha(file)
    for case in all_cases:
        file = ROOT/f"runs/s6_pin_text_v1/predictions/{case}/pin_text_result.json"
        protected[str(file)] = sha(file)
    protocol = dict(status="presets_fixed_before_development_comparison", presets=PRESETS, groups=groups,
        tuning_variant="reviewed_v7", selection="highest development micro endpoint F1; fixed preset order breaks ties",
        independent_test=False, check_group_note="Historical validation group already inspected in past work; not a fresh test",
        evaluator="global one-to-one two-endpoint Euclidean<=5px; no net conditioning; draft references",
        original_predictions_modified=False, component_guard="existing image-hash-verified detection predictions, bridge-only", official_NetLineF1=None, protected_sha256=protected)
    write(OUT/"protocol.json", protocol)
    candidates = {}
    for name, cfg in PRESETS.items():
        rows = []
        for case in groups["development"]:
            _, record = decode_case("reviewed_v7", case, cfg, by_variant["reviewed_v7"][case]["segments"])
            rows.append(record)
        candidates[name] = dict(summary=total(rows), cases=rows)
        print(json.dumps(dict(preset=name, development=candidates[name]["summary"])), flush=True)
    selected = max(PRESETS, key=lambda n: candidates[n]["summary"]["f1"])
    cfg = PRESETS[selected]
    write(OUT/"selection.json", dict(selected_preset=selected, config=cfg, development_only=True, candidates=candidates))
    write(ROOT/"configs/wire_segments_v1.json", cfg | dict(version="wire-segments-v1", selected_preset=selected,
        selection_record=str(OUT/"selection.json"), official_NetLineF1=None))
    comparisons = {}
    for variant in by_variant:
        rows = []; originals = []
        for case in all_cases:
            reference = by_variant[variant][case]
            decoded, record = decode_case(variant, case, cfg, reference["segments"])
            folder = OUT/variant/case
            write(folder/"segments.json", decoded)
            write(folder/"candidate_edges.json", export_candidate_edges(decoded))
            write(folder/"evaluation.json", record)
            rows.append(record)
            originals.append(dict(case_id=case, metrics=reference["decoded_endpoint_proxy"]))
            print(json.dumps(dict(variant=variant, case=case, f1=record["metrics"]["f1"], merges=record["merged_gaps"])), flush=True)
            if variant=="reviewed_v7" and case in {"0007", "0036", "0208", "0029"}:
                render(case, reference["decoded_segments"], [r["points"] for r in decoded["segments"]], reference["segments"])
        results = {}
        for group, members in dict(all26=all_cases, **groups).items():
            results[group] = dict(before=total([r for r in originals if r["case_id"] in members]),
                                  after=total([r for r in rows if r["case_id"] in members]))
        comparisons[variant] = dict(groups=results, cases=rows)
    assert all(sha(path)==digest for path, digest in protected.items())
    result = dict(status="completed", selected_preset=selected, config=cfg, comparisons=comparisons,
                  official_NetLineF1=None, independent_test=False, original_inputs_unchanged=True,
                  default_weights_replaced=False, training_performed=False)
    write(OUT/"comparison.json", result)
    lines = ["# 两点线段解码实验", "", "冻结分割模型与0.5阈值，改进的是mask→几何解码。端点按5px方向无关、一对一全图匹配；没有net归属限制，参考未认证，不是官方NetLineF1。", "",
        f"固定3套候选，先在既有13页development上按端点F1选择{selected}，再检查另13页。另一组曾用于历史验证和本轮前的诊断，不能称独立测试。", "",
        "| 模型 | 页面 | 原端点F1 | 新端点F1 | 原TP/FP/FN | 新TP/FP/FN | 合并缺口数 |",
        "|---|---|---:|---:|---|---|---:|"]
    for variant, data in comparisons.items():
        for group, scores in data["groups"].items():
            a, b = scores["before"], scores["after"]
            lines.append(f"| {variant} | {group} | {a['f1']:.2%} | {b['f1']:.2%} | {a['tp']}/{a['fp']}/{a['fn']} | {b['tp']}/{b['fp']}/{b['fn']} | {b['merged_gaps']} |")
    lines += ["", "实现：保留折点与分叉位置，删除≤2px的短毛刺，对分段像素拟合直线并对齐近水平/垂直方向；仅在固定距离、方向、横向偏差、沿线支持、预测pin和器件主体保护条件下合并碎段。原像素掩码不修改。组件保护仅来自已有检测预测并核对输入图像哈希，不使用标注框；仅限制新增跨缺口向量，不修改原mask。", "",
        f"配置：{json.dumps(cfg, ensure_ascii=False)}", "",
        "segments.json记录内部左上坐标、每段源碎段和合并缺口；candidate_edges.json提供左下H-y坐标的两点edges候选。它们尚未归网，不能直接冒充完整参赛nets。网络连接及零长度逻辑飞线仍由后续网络层处理。", "",
        "评价中分别记录参考几何无响应、部分响应和有充分响应但没匹配成线段；未匹配预测中还统计可能的截短/多分段。这些只是对草稿几何的诊断提示，参考缺失/分段歧义仍可能影响计数。", "",
        "0208缺少模型响应的限制没有靠几何解码消失。所有输入预测、权重、参考哈希保持不变，未重新训练，也未替换默认权重。", "",
        "![0007](comparison_0007.png)", "", "![0036](comparison_0036.png)", "", "![0208](comparison_0208.png)"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(selected_preset=selected, comparisons={v:d['groups'] for v,d in comparisons.items()}), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
