"""Inspect remaining missing paths; never infer or apply bulk wire labels."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.dataset_converter import load_source_annotation
from pcb_parser.unet_data import point, polylines

OUT = ROOT / "runs/unet_missing_path_audit_20261015"
LABELS = {
    "straight_ink_supported": "连续水平/竖直坐标，且至少95%坐标有墨迹：路径恢复候选",
    "straight_weak_ink": "连续直线坐标，但原图墨迹支持不足：需视觉核查",
    "ordered_nonstraight": "坐标按8邻域连续排列，非单一直线：需核对转折与语义",
    "two_sparse_coordinates": "coordinates只保存两个不相邻点：完整路线未知，不能直接连线",
    "other_coordinates": "有坐标，但不满足连续直线或有序路径条件：需进一步处理",
    "one_point": "仅一个坐标：不能据此恢复线段",
    "invalid_or_out_of_bounds": "坐标含无效、非整数或越界点：需核查坐标约定",
    "skeleton_only": "无coordinates，但有骨架坐标：需核查替代几何",
    "endpoints_only": "只有端点：不能直接假定两点间为直线",
    "no_geometry": "没有可用坐标、骨架或端点：需回原图判定或重标",
}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def points(value):
    if not isinstance(value, (list, tuple)):
        return []
    return [p for item in value if (p := point(item)) is not None]


def classify(edge, width, height, rgb):
    raw = edge.get("coordinates", [])
    coords = points(raw)
    keys = edge.get("keyPoints", {})
    skeleton = points(keys.get("skeleton", {}).get("wire", []))
    endpoints = points(keys.get("endPoints", {}).get("wire", []))
    info = dict(raw_coordinate_count=len(raw) if isinstance(raw, list) else 0,
                valid_coordinate_count=len(coords), skeleton_count=len(skeleton),
                endpoint_count=len(endpoints), bbox=None, ink_fraction=None,
                source_8connected_sequence=False, unique_coordinate_count=0)
    if coords:
        info["bbox"] = [min(p[0] for p in coords), min(p[1] for p in coords),
                        max(p[0] for p in coords), max(p[1] for p in coords)]
        if len(coords) != info["raw_coordinate_count"] or any(
            x != round(x) or y != round(y) or not (0 <= x < width and 0 <= y < height)
            for x, y in coords
        ):
            return "invalid_or_out_of_bounds", info
        unique = sorted(set(coords))
        info["unique_coordinate_count"] = len(unique)
        info["ink_fraction"] = float(np.mean([np.min(rgb[int(y), int(x)]) < 224 for x, y in unique]))
        if len(unique) < 2:
            return "one_point", info
        xs, ys = [p[0] for p in unique], [p[1] for p in unique]
        straight = ((len(set(ys)) == 1 and len(unique) == int(max(xs) - min(xs)) + 1)
                    or (len(set(xs)) == 1 and len(unique) == int(max(ys) - min(ys)) + 1))
        if straight:
            return ("straight_ink_supported" if info["ink_fraction"] >= .95 else "straight_weak_ink"), info
        if len(unique) == 2 and max(abs(unique[1][0] - unique[0][0]), abs(unique[1][1] - unique[0][1])) > 1:
            return "two_sparse_coordinates", info
        adjacent = all(max(abs(b[0] - a[0]), abs(b[1] - a[1])) <= 1 for a, b in zip(coords, coords[1:]))
        info["source_8connected_sequence"] = adjacent
        return ("ordered_nonstraight" if adjacent else "other_coordinates"), info
    if skeleton:
        return "skeleton_only", info
    if endpoints:
        return "endpoints_only", info
    return "no_geometry", info


def main():
    base_manifest = ROOT / "data/segmentation_unet_v1/manifest.jsonl"
    derived_manifest = ROOT / "data/segmentation_unet_v3_local_path_repairs/manifest.jsonl"
    base = rows(base_manifest)
    derived = rows(derived_manifest)
    raw = {r["case_id"]: r for r in rows(ROOT / "data/manifest.jsonl")}
    base_missing = [r for r in base if r["stats"]["missing_physical_paths"]]
    remaining = [r for r in derived if r["stats"]["missing_physical_paths"]]
    assert len(base_missing) == 122 and len(remaining) == 121
    assert sum(r["stats"]["missing_physical_paths"] for r in base_missing) == 692
    assert sum(r["stats"]["missing_physical_paths"] for r in remaining) == 691
    frozen = {str(p): sha(p) for p in (base_manifest, derived_manifest, ROOT / "data/manifest.jsonl")}
    records = []
    for index, row in enumerate(remaining, 1):
        source = Path(raw[row["case_id"]]["raw_annotation_path"])
        frozen[str(source)] = sha(source)
        annotation, _ = load_source_annotation(source)
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"))
        found = 0
        for eid, edge in annotation.get("edges", {}).items():
            if edge.get("state") == "deleted" or str(edge.get("isFlywire")).lower() == "true" or edge.get("category", "wire") != "wire":
                continue
            paths = edge.get("keyPointPaths", {})
            if list(polylines(paths.get("wire", []) if isinstance(paths, dict) else paths)):
                continue
            category, info = classify(edge, row["width"], row["height"], rgb)
            records.append(dict(case_id=row["case_id"], split=row["split"], edge_id=eid,
                                category=category, meaning=LABELS[category],
                                image=row["image"], source=str(source), source_pointer=f"/edges/{eid}",
                                connect=edge.get("connect", {}), **info))
            found += 1
        assert found == row["stats"]["missing_physical_paths"], row["case_id"]
        if index % 30 == 0:
            print(json.dumps(dict(phase="source_geometry_audit",pages=index,total=len(remaining)),ensure_ascii=False),flush=True)
    assert len(records) == 691
    counts = Counter(r["category"] for r in records)
    by_page = defaultdict(list)
    for record in records:
        by_page[record["case_id"]].append(record)
    all_straight = [case for case, items in by_page.items() if all(r["category"] == "straight_ink_supported" for r in items)]
    all_straight_set = set(all_straight)
    continuous_categories = {"straight_ink_supported", "ordered_nonstraight"}
    continuous = [r for r in records if r["category"] in continuous_categories and r["ink_fraction"] >= .95]
    all_continuous_pages = [case for case, items in by_page.items() if all(r["category"] in continuous_categories and r["ink_fraction"] >= .95 for r in items)]
    no_geom_pages = [case for case, items in by_page.items() if any(r["category"] in {"no_geometry", "endpoints_only", "one_point", "two_sparse_coordinates"} for r in items)]
    assert all(sha(p) == digest for p, digest in frozen.items())
    OUT.mkdir(exist_ok=True)
    summary = dict(
        status="remaining_missing_path_audit_complete_no_bulk_repairs",
        baseline_missing_pages=122, baseline_missing_edges=692,
        fixed_case="0005", fixed_edges=1,
        remaining_missing_pages=121, remaining_missing_edges=691,
        category_edge_counts=dict(counts),
        pages_all_missing_edges_straight_ink_supported=len(all_straight),
        all_straight_pages=all_straight,
        all_straight_pages_by_split=dict(Counter(r["split"] for r in remaining if r["case_id"] in all_straight_set)),
        all_straight_pages_with_other_structural_flags=[r["case_id"] for r in remaining if r["case_id"] in all_straight_set and any(r["stats"][k] for k in ("out_of_bounds_paths", "invalid_pin_tips", "unowned_root_pins"))],
        pages_any_insufficient_geometry=len(no_geom_pages),
        pages_insufficient_geometry=no_geom_pages,
        continuous_ink_supported_edges=len(continuous),
        continuous_ink_supported_fraction=len(continuous)/len(records),
        pages_all_missing_edges_continuous_ink_supported=len(all_continuous_pages),
        all_continuous_pages_by_split=dict(Counter(r["split"] for r in remaining if r["case_id"] in set(all_continuous_pages))),
        ink_support_by_category={k:dict(records=len(v),ink_support_ge95=sum(r["ink_fraction"]>=.95 for r in v),min_ink=min(r["ink_fraction"] for r in v)) for k in counts if (v:=[r for r in records if r["category"]==k])},
        category_definitions=LABELS,
        method="Inspect active non-flywire edges with empty parsed keyPointPaths; validate coordinates, continuous horizontal/vertical samples, source sequence and original-image ink at min(R,G,B)<224.",
        limitations=["Image ink is not semantic wire certification; no automatic page approval or noise-rate estimate.",
                     "Coordinates can represent drawing fragments or pin stems; wire/pin conventions and active-record validity still require visual review.",
                     "Only declared missing paths are audited; invisible missing source records and existing incorrect paths are outside scope."],
        source_files_unchanged=True, dataset_manifests_unchanged=True,
        training_started=False, repaired_during_this_audit=0,
        source_sha256=frozen,
    )
    (OUT / "audit.json").write_text(json.dumps(dict(summary=summary, records=records),ensure_ascii=False,indent=2),encoding="utf-8")
    with (OUT / "records.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        fields=["case_id","split","edge_id","category","meaning","raw_coordinate_count","valid_coordinate_count","unique_coordinate_count","skeleton_count","endpoint_count","bbox","ink_fraction","source"]
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(records)
    lines=["# 剩余缺路径记录审查 · 2026-10-15", "", "0005 已单独修订后，剩余 **121页、691条缺路径记录**。本次仅审查，没有批量补标。", "", "| 记录类别 | 条数 |", "|---|---:|"]
    lines.extend(f"| {LABELS[k]} | {counts[k]} |" for k in LABELS if counts[k])
    lines += ["", f"连续坐标且原图墨迹支持≥95%的记录：{len(continuous)}/691；全部缺路径记录都满足这一条件的页面：{len(all_continuous_pages)}页；划分：{summary['all_continuous_pages_by_split']}。这些仅是路径恢复/视觉复核候选，不能直接称为已认证训练页。", "", f"其中全部为连续水平/竖直线的页面：{len(all_straight)}页。", "", f"缺乏完整路线几何的页面：{len(no_geom_pages)}页，案例为{no_geom_pages}；当前均是coordinates仅有两个不相邻点。", "", "## 各类记录例子", ""]
    for category in LABELS:
        examples=[r for r in records if r["category"]==category][:3]
        if examples:
            lines += [f"- {LABELS[category]}："+"；".join(f"{r['case_id']}/{r['edge_id']}，coordinates={r['valid_coordinate_count']}点，bbox={r['bbox']}" for r in examples)]
    lines += ["", "图像墨迹不等于导线语义；本次不认证wire/pin分界、器件内部线、接点或网络连接。源标注、训练/验证清单均未修改，未训练模型。", "", "完整逐记录数据：audit.json / records.csv。"]
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"source_sha256","category_definitions","all_straight_pages","limitations","method"}},ensure_ascii=False))


if __name__ == "__main__":
    main()
