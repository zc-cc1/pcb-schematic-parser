"""Offline geometry audit of frozen U-Net masks; never require path connectivity.

References are used only for scoring. Skeleton decoding receives the model mask
alone. Global endpoint matches are a diagnostic, not official net-conditioned F1.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import polylines
from pcb_parser.wire_graph import compress_skeleton
from verify_unet_contrast import path_specs, path_records

OUT = ROOT/"runs/unet_competition_requirements_20261015"
EVALUATION = ROOT/"runs/training/unet_v7_finetune_comparison_20261015/evaluation"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_segments(annotation):
    """Unique, nonzero adjacent source-point pairs; no 10px length exclusion."""
    result = []
    seen = set()
    for eid, edge in annotation.get("edges", {}).items():
        if edge.get("state") == "deleted" or str(edge.get("isFlywire")).lower() == "true":
            continue
        if edge.get("category", "wire") != "wire":
            continue
        raw = edge.get("keyPointPaths", {})
        raw = raw.get("wire", []) if isinstance(raw, dict) else raw
        for path in polylines(raw):
            for a, b in zip(path, path[1:]):
                points = (tuple(map(float, a)), tuple(map(float, b)))
                key = min(points, points[::-1])
                if key in seen or math.dist(a, b) <= 0:
                    continue
                seen.add(key)
                result.append(dict(edge_id=eid, points=points, length=math.dist(a, b)))
    return result


def evidence(mask, references):
    """Presence near reference geometry, with no connected-component condition."""
    y, x = np.where(mask)
    tree = cKDTree(np.column_stack((x, y))) if len(x) else None
    records = []
    for segment in references:
        a, b = segment["points"]
        samples = np.linspace(a, b, max(2, math.ceil(segment["length"])+1))
        distance = tree.query(samples)[0] if tree is not None else np.full(len(samples), np.inf)
        row = dict(segment, sample_count=len(samples))
        for radius in (2, 5):
            close = distance <= radius
            row[f"coverage_{radius}px"] = float(close.mean())
            row[f"both_ends_have_pixels_{radius}px"] = bool(close[0] and close[-1])
            row[f"sample_hits_{radius}px"] = int(close.sum())
        row["supported_90pct_and_ends_5px"] = bool(row["coverage_5px"] >= .9 and row["both_ends_have_pixels_5px"])
        row["almost_absent_5px"] = bool(row["coverage_5px"] < .1)
        records.append(row)
    return records


def decode_mask(mask):
    """Existing decoder, unchanged; no annotation or reference points supplied."""
    skeleton = skeletonize(mask, method="zhang")
    graph = compress_skeleton(skeleton, mask.astype(np.float32), {"polyline_tolerance_px": 1})
    segments = []
    seen = set()
    for _, _, data in graph.edges(data=True):
        for a, b in zip(data["polyline"], data["polyline"][1:]):
            points = (tuple(a), tuple(b))
            key = min(points, points[::-1])
            if key not in seen and math.dist(a, b) > 0:
                seen.add(key); segments.append(points)
    return segments


def match_segments(predictions, references):
    """One-to-one endpoint match <=5px, orientation independent, without nets."""
    n, m = len(predictions), len(references)
    matches = []
    if n and m:
        p = np.asarray(predictions)
        r = np.asarray([s["points"] for s in references])
        forward = np.maximum(np.linalg.norm(p[:, None, 0]-r[None, :, 0], axis=-1),
                             np.linalg.norm(p[:, None, 1]-r[None, :, 1], axis=-1))
        reverse = np.maximum(np.linalg.norm(p[:, None, 0]-r[None, :, 1], axis=-1),
                             np.linalg.norm(p[:, None, 1]-r[None, :, 0], axis=-1))
        errors = np.minimum(forward, reverse)
        penalty = (min(n, m)+1)*6
        pi, ri = linear_sum_assignment(np.where(errors <= 5, errors, penalty))
        matches = [dict(prediction=int(i), reference=int(j), max_endpoint_error=float(errors[i, j]))
                   for i, j in zip(pi, ri) if errors[i, j] <= 5]
    tp = len(matches)
    return dict(tp=tp, fp=n-tp, fn=m-tp, predictions=n, references=m,
                precision=tp/max(1, n), recall=tp/max(1, m), f1=2*tp/max(1, n+m), matches=matches)


def aggregate(records):
    segments = [s for row in records for s in row["segments"]]
    n = len(segments)
    totals = {k: sum(row["decoded_endpoint_proxy"][k] for row in records) for k in ("tp", "fp", "fn")}
    tp, fp, fn = (totals[k] for k in ("tp", "fp", "fn"))
    proxy = dict(totals, precision=tp/max(1, tp+fp), recall=tp/max(1, tp+fn), f1=2*tp/max(1, 2*tp+fp+fn))
    return dict(pages=len(records), reference_segments=n,
        both_ends_have_pixels_5px=sum(s["both_ends_have_pixels_5px"] for s in segments),
        supported_90pct_and_ends_5px=sum(s["supported_90pct_and_ends_5px"] for s in segments),
        almost_absent_5px=sum(s["almost_absent_5px"] for s in segments),
        sampled_length_coverage_5px=sum(s["sample_hits_5px"] for s in segments)/max(1, sum(s["sample_count"] for s in segments)),
        decoded_endpoint_proxy=proxy)


def self_check():
    mask = np.zeros((20, 50), bool); mask[10, 5:46] = True; mask[10, 24:27] = False
    references = [dict(edge_id="test", points=((5, 10), (45, 10)), length=40)]
    assert evidence(mask, references)[0]["supported_90pct_and_ends_5px"]
    spec = dict(edge_id="test", points=((5, 10), (45, 10)), length=40,
                tube=np.zeros_like(mask))
    spec["tube"][8:13, 5:46] = True
    assert not path_records(mask, [spec])[0]["connected_within_reference_tube"]
    assert evidence(np.zeros_like(mask), references)[0]["almost_absent_5px"]
    matched = match_segments([((45, 10), (5, 10)), ((5, 10), (45, 10))], references)
    assert (matched["tp"], matched["fp"], matched["fn"]) == (1, 1, 0)


def render_example(case, variants, references, roi, title):
    image = np.asarray(Image.open(ROOT/f"data/standardized/{case}/{case}.png").convert("RGB"))
    x0, y0, x1, y1 = roi
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), dpi=150, layout="constrained")
    for ax, name in zip(axes, (None, "baseline_v6", "reviewed_v7")):
        panel = image.copy()
        if name:
            mask = np.asarray(Image.open(EVALUATION/name/case/"wire_mask.png")) > 0
            panel[mask] = (.15*panel[mask]+.85*np.array([0, 170, 95])).astype(np.uint8)
        ax.imshow(panel[y0:y1, x0:x1], interpolation="nearest")
        if name is None:
            for ref in references:
                a, b = np.asarray(ref["points"])
                ax.plot([a[0]-x0, b[0]-x0], [a[1]-y0, b[1]-y0], color="#df5b19", lw=1, alpha=.7)
        ax.set_xlim(0, x1-x0); ax.set_ylim(y1-y0, 0); ax.axis("off")
        ax.set_title("原图 + JSON参考线（橙色）" if name is None else name+"：绿色为模型预测")
    fig.suptitle(title)
    path = OUT/f"geometry_example_{case}.png"
    fig.savefig(path, facecolor="white"); plt.close(fig)


def endpoint_examples(records):
    examples = []
    for row in records:
        matched = {m["reference"]: m for m in row["decoded_endpoint_proxy"]["matches"]}
        for path in row["old_connectivity_failures_with_90pct_geometric_support"]:
            if len(path["points"]) != 2:
                continue
            key = sorted(map(tuple, path["points"]))
            for index, segment in enumerate(row["segments"]):
                rounded = sorted(tuple(round(v) for v in point) for point in segment["points"])
                if rounded == key and index in matched:
                    match = matched[index]
                    examples.append(dict(case_id=row["case_id"], source_path=path,
                        decoded_segment=row["decoded_segments"][match["prediction"]],
                        max_endpoint_error=match["max_endpoint_error"]))
    return examples


def write_report(result):
    summaries = result["summaries"]
    lines = ["# U-Net输出与0803Rev1导线要求核对", "",
        "结论：导线几何无需逐像素连续；当前模型能为大量线段提供几何证据，但尚不能表达所有导线。原2px路径连续性指标不能单独决定模型采用。没有替换默认权重或再次训练。", "",
        "## 原文核对（PDF页码）", "",
        "来源为用户提供的30页0803Rev1，原件路径及SHA256见source_manifest.json，原件未修改。", "",
        "- 第6页：NetLineF1权重0.10，NetHypergraphF1权重0.35。此前把线段指标称为0.35是错误的。",
        "- 第8页：仅在matched net pair内比较线段，两个端点默认5px容差；没有像素IoU或像素连通硬性要求。",
        "- 第20页：每个edge恰好两个点；折线按折点拆分；多分/少分都会扣分，允许零长度线段。",
        "- 第23页：飞线的逻辑关联不能靠补一条远距离物理导线表达；示例使用点状edges。",
        "- 第30页：线段放入错误net不能因几何接近而得分。因而放松像素连续性不等于忽略网络成员关系。", "",
        "## 对26页已有预测的重新检查", "",
        "复用相同26页、三个冻结模型的0.5阈值预测掩码，没有重训。参考为源JSON中非deleted、非flywire的物理wire路径，逐相邻点拆成两点段、全页方向无关去重，保留短线段，排除零长度段，共2266段。这不是此前1548条≥10px折线路径，也不是官方Golden线段数。", "",
        "几何证据定义：沿每段以不超过1px间隔采样，至少90%采样位置和两个端点附近5px内有预测像素；不要求这些像素互相连通。端点附近有像素不等于解码端点位置正确。", "",
        "| 模型 | 沿线采样覆盖（5px） | 有充分几何证据 | 几乎无响应的参考段 | 简单解码后的全图端点匹配F1 |",
        "|---|---:|---:|---:|---:|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['sampled_length_coverage_5px']:.2%} | {s['supported_90pct_and_ends_5px']}/{s['reference_segments']} | {s['almost_absent_5px']} | {s['decoded_endpoint_proxy']['f1']:.2%} |")
    lines += ["", "几乎无响应指5px邻域覆盖不足10%。两个覆盖统计不是正式召回；5px邻域可能包含附近其他线路、引脚或误检。草稿参考也有漏标、错位和分段歧义，不能把所有失败直接归为模型漏线。", "",
        "解码使用现有compress_skeleton（Zhang骨架＋RDP容差1px），仅输入预测mask；没有组件遮罩、断线合并或网络分组。随后在全图按两个端点最大距离≤5px、方向无关、一对一匈牙利匹配。表中F1是跨页累计的本地几何代理：没有matched net限制、没有正式Golden、没有完整官方边界规则，不是NetLineF1，也不是模型可达到的性能上限。其较低值同时受碎段、额外分段、端点偏移、噪声及参考分段差异影响。", "",
        "## 原连续性指标会否低估可表达的线段", ""]
    for name, examples in result["strict_fail_but_decoded_endpoint_match"].items():
        lines.append(f"- {name}：{len(examples)}条原严格检查失败的单段路径，已有mask-only解码结果通过两个端点5px几何匹配。")
    lines += ["", "例：0007/E_17，修订组参考(156,160)→(218,160)，解码(158,162)→(217,160)，两个端点误差约2.83px与1px；原2px路径检查不通过，但满足当前5px几何代理。这不证明net归属也正确。", "",
        "## 明确不能仅靠换表示方式恢复的例子", "",
        "0208是灰色导线原图。修订模型整页仅51个正像素，66段源参考中65段几乎无响应，简单解码0段端点匹配成功。可见导线大片缺失，不能仅从当前阈值掩码可靠拟合出所有线段。没有据此断言概率图、原始图像或改进模型也无法恢复。", "",
        "![0208灰线漏检](geometry_example_0208.png)", "",
        "## 依据赛题应如何推进", "",
        "1. 保留原路径连续性为诊断，取消其作为模型采用的单独否决条件。原自动比较程序推荐baseline_v6是按旧门槛得出的历史结果，不能解释为新模型不适合赛题。",
        "2. 优先验证图像/模型驱动的线段拟合、短缺口跨越、同一直线碎段合并及折点定位；最终按参考分段口径输出两个端点，控制多余/缺少线段。不能在推理时照着参考JSON补线。",
        "3. 用5px一对一端点匹配评价几何解码，单列净漏检、误检、过度分段与端点偏差；网络身份可靠后再按matched net pair评估。",
        "4. 0208这类缺少导线响应的样本需改进图像预处理、训练覆盖/增强或模型；分割IoU与骨架连续性都不是唯一目标。",
        "5. hyperGraph的pin set与线段所属net仍需正确。当前裸U-Net输出概率图/掩码与pin-tip候选，不直接给出完整edge集合或net成员。", "",
        "原文页渲染已人工视觉核对6、8、19、20、23页，另核对第30页文字；geometry_audit.json保存逐段证据、实际解码线段、一对一匹配、严格失败但端点匹配的实例及输入保护哈希。全部保护输入SHA256保持不变。"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    self_check(); OUT.mkdir(parents=True, exist_ok=True)
    comparison = read(EVALUATION/"comparison.json")
    raw = {row["case_id"]: row for row in map(json.loads, (ROOT/"data/manifest.jsonl").read_text(encoding="utf-8").splitlines())}
    cases = [r["case_id"] for r in comparison["variants"]["baseline_v6"]["cases"] if r["unique26"]]
    protected = {str(EVALUATION/"comparison.json"): sha(EVALUATION/"comparison.json")}
    variants = {name: [] for name in comparison["variants"]}
    annotations = {}
    for case in cases:
        annotation, _ = load_source_annotation(raw[case]["raw_annotation_path"])
        refs = reference_segments(annotation)
        annotations[case] = refs
        protected[raw[case]["raw_annotation_path"]] = sha(raw[case]["raw_annotation_path"])
        for name, records in variants.items():
            mask_file = EVALUATION/name/case/"wire_mask.png"
            protected[str(mask_file)] = sha(mask_file)
            mask = np.asarray(Image.open(mask_file)) > 0
            segments = evidence(mask, refs)
            decoded = decode_mask(mask)
            proxy = match_segments(decoded, refs)
            paths = path_records(mask, path_specs(annotation, mask.shape))
            # Expose paths failing the old strict diagnostic despite >=90% geometric support.
            relaxed_failures = []
            for path in paths:
                if path["connected_within_reference_tube"]:
                    continue
                parts = [dict(edge_id=path["edge_id"], points=(a, b), length=math.dist(a, b))
                         for a, b in zip(path["points"], path["points"][1:]) if math.dist(a, b) > 0]
                support = evidence(mask, parts)
                if support and all(s["supported_90pct_and_ends_5px"] for s in support):
                    relaxed_failures.append(dict(edge_id=path["edge_id"], points=path["points"], length=path["length"]))
            record = dict(case_id=case, foreground_pixels=int(mask.sum()), segments=segments,
                          decoded_endpoint_proxy=proxy, decoded_segments=decoded,
                          old_connectivity_failures_with_90pct_geometric_support=relaxed_failures)
            records.append(record)
            print(json.dumps(dict(case=case, variant=name, references=len(refs),
                supported=sum(s["supported_90pct_and_ends_5px"] for s in segments),
                absent=sum(s["almost_absent_5px"] for s in segments), decoded_matches=proxy["tp"])), flush=True)
    summary = {name: aggregate(records) for name, records in variants.items()}
    result = dict(status="completed_offline_geometry_diagnostic", reference_status="original_source_geometry_not_official_Golden",
        official_NetLineF1=None, official_net_matching_used=False, topology_required_for_evidence=False,
        mask_threshold=.5, endpoint_tolerance_px=5, reference_segments="deduplicated nonzero adjacent source points; flywires excluded; short segments included",
        supported_definition="at least 90% of <=1px-spaced reference samples and both ends have a predicted pixel within 5px; NOT decoded endpoint accuracy",
        endpoint_proxy_definition="existing mask-only skeleton decoder/RDP tolerance1; orientation-independent one-to-one global endpoint match<=5px; no net conditioning, no gap merging",
        scope="development validation only; geometry presence is not proof of correct output segments or net membership",
        summaries=summary, variants=variants, protected_sha256=protected,
        strict_fail_but_decoded_endpoint_match={name: endpoint_examples(records) for name, records in variants.items()})
    assert all(sha(p) == h for p, h in protected.items())
    result["protected_inputs_unchanged"] = True
    (OUT/"geometry_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    render_example("0208", variants, annotations["0208"], (405, 115, 715, 280),
                   "0208 · 灰色导线缺少模型响应；端点表示也不能凭空恢复全部几何")
    render_example("0036", variants, annotations["0036"], (535, 30, 790, 120),
                   "0036 · 短缺口不必直接判成整条导线不可表达；仍需正确拟合端点")
    write_report(result)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
