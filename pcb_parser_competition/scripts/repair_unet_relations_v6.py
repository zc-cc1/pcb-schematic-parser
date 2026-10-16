"""Locally reannotate ten false associations from image evidence; keep reversible edits."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_parser.unet_data import annotation_targets, polylines
from repair_unet_0005_dp_path import read_rows, sha, usable, write_json

BASE = ROOT / "data/segmentation_unet_v5_reviewed_supervision_20261015"
OUT = ROOT / "data/segmentation_unet_v6_corrected_relations_20261015"
REPORT = ROOT / "runs/unet_relation_repair_v6"
REASONS = {
    "0101": {"E_157": "3V3错误跨接C3上端；3V3应加入C3下端/VOUT的E_62，上端接原GND支路。"},
    "0102": {"E_223": "C34误标为竖直电容，并错误跨接GPIO2；纠正为水平电容，GPIO2恢复自身短线。",
             "E_224": "C34另一端错误跨接QSPI_SS；应接旁边GND，QSPI_SS恢复自身水平短线。"},
    "0124": {"E_103": "C8上端横坐标偏移10px并错接PB9；上端应接GND，PB9有明确未连接叉号。"},
    "0150": {"E_206": "USB DP2错误接到下方错位GND；DP2恢复水平标号线，GND接壳体及B1A12。",
             "E_207": "USB CC1错误接到下方GND附近；恢复CC1自身水平标号线。"},
    "0180": {"E_31": "RD/P3.7没有外部可见连线，撤销指向下方空白的假关联，保留引脚。",
             "E_32": "WR/P3.6没有外部可见连线，撤销指向下方接地附近的假关联，保留引脚。",
             "E_33": "VCC应接右侧40脚；左侧P1.0应加入上方已有E_3，不能跨芯片相连。"},
    "0238": {"E_43": "C3接地与DREG是不同支路；恢复C3下端接地及DREG到其独立电容的线路。"},
}
ROIS = {
    "0101": [(135, 285, 220, 365)],
    "0102": [(1935, 240, 2095, 330), (1545, 430, 1618, 465), (2060, 460, 2130, 490)],
    "0124": [(1045, 40, 1125, 140)],
    "0150": [(110, 88, 165, 165), (105, 190, 145, 285)],
    "0180": [(530, 230, 710, 275), (528, 397, 557, 490)],
    "0238": [(340, 128, 392, 195), (118, 280, 192, 370)],
}


class Editor:
    def __init__(self, case, original):
        self.case = case
        self.data = copy.deepcopy(original)
        self.operations = []
        self.routes = []
        self.retired = []
        self.changed_pins = set()
        self.new_pins = []

    def note(self, reason):
        return dict(reason=reason, reviewer="agent", evidence="original_png_visual_review",
                    independent_human_certification=False, full_page_certified=False)

    def pin(self, cid, pid, **changes):
        for p in (self.data["pins"][pid], self.data["components"][cid]["pins"][pid]):
            p.update(copy.deepcopy(changes))
        self.changed_pins.add((cid, pid))

    def new_pin(self, cid, tip, base, name=None, number=None):
        occupied = set(self.data["pins"])
        occupied.update(pid for c in self.data["components"].values() for pid in c.get("pins", {}))
        pid = f"P_{max(int(p[2:]) for p in occupied) + 1}"
        pin = dict(pinId=pid, edgeId=None, pinNumberByOCR=None, pinNumberByVLM=None,
                   pinNameByOCR=None, pinNameByVLM=None, tip=tip, base=base,
                   manual_relation_review=self.note("补充原图清晰可见但源记录缺少的器件引脚"))
        if name is not None: pin["pinNameByReview"] = name
        if number is not None: pin["pinNumberByReview"] = number
        self.data["pins"][pid] = copy.deepcopy(pin)
        self.data["components"][cid].setdefault("pins", {})[pid] = copy.deepcopy(pin)
        self.changed_pins.add((cid, pid)); self.new_pins.append((cid, pid))
        return pid

    def component(self, cid, bbox=None, **changes):
        c = self.data["components"][cid]
        c.update(changes)
        if bbox:
            c["realBBox"] = dict(zip(("x1", "y1", "x2", "y2"), bbox))
            c["location"] = copy.deepcopy(c["realBBox"])

    def retire(self, eid, reason):
        e = self.data["edges"][eid]
        assert e["state"] == "active"
        e["state"] = "deleted"
        e.pop("wire_supervision", None)
        e["manual_relation_review"] = self.note(reason)
        self.retired.append(eid)
        for pid, connection in e.get("connect", {}).items():
            cid = connection["cpntId"]
            if self.data["pins"][pid].get("edgeId") == eid:
                self.pin(cid, pid, edgeId=None)
        self.operations.append(dict(action="retire_false_association", edge_id=eid, reason=reason))

    def edge(self, eid, paths, connections, reason):
        if eid is None:
            eid = f"E_{max(int(e[2:]) for e in self.data['edges']) + 1}"
            self.data["edges"][eid] = dict(edgeId=eid, netlabel="", text={}, location=[], associatedTexts={})
            action = "add_visible_connection"
        else:
            action = "restore_deleted_visible_route" if self.data["edges"][eid]["state"] == "deleted" else "correct_visible_route_and_membership"
        edge = self.data["edges"][eid]
        old_connections = copy.deepcopy(edge.get("connect", {}))
        for pid, conn in old_connections.items():
            if pid not in {p for _, p in connections} and self.data["pins"][pid].get("edgeId") == eid:
                self.pin(conn["cpntId"], pid, edgeId=None)
        coords = []
        seen = set()
        for path in paths:
            for a, b in zip(path, path[1:]):
                assert a[0] == b[0] or a[1] == b[1], "Only image-reviewed orthogonal routes"
                length = int(abs(b[0]-a[0]) + abs(b[1]-a[1]))
                for k in range(length+1):
                    t = k / length if length else 0
                    p = (round(a[0]+(b[0]-a[0])*t), round(a[1]+(b[1]-a[1])*t))
                    if p not in seen: coords.append(list(p)); seen.add(p)
        conn = {}
        for cid, pid in connections:
            self.pin(cid, pid, edgeId=eid)
            conn[pid] = dict(pinId=pid, cpntId=cid, location=self.data["pins"][pid]["tip"])
        edge.update(state="active", category="wire", isFlywire=False, connect=conn,
                    keyPointPaths={"wire": paths}, coordinates=coords, originalCoordinatesCount=len(coords),
                    keyPoints={"skeleton": {"wire": coords}, "endPoints": {"wire": [p for line in paths for p in (line[0], line[-1])]},
                               "turnPoints": {"wire": [p for line in paths for p in line[1:-1]]}, "divergePoints": {"wire": []}},
                    manual_relation_review=self.note(reason))
        edge.pop("wire_supervision", None)
        self.routes.append(eid)
        self.operations.append(dict(action=action, edge_id=eid, reason=reason))
        return eid


def reannotate(editor):
    e = editor
    for eid, reason in REASONS[e.case].items(): e.retire(eid, reason)
    if e.case == "0101":
        e.component("C_28", [148, 344, 150, 355], rotation=90)
        e.pin("C_28", "P_242", tip=[150,349], base=[149,349])
        e.pin("C_38", "P_70", tip=[160,350], base=[160,349])
        e.edge("E_62", [[[150,349],[160,349],[160,355],[211,355]]],
               [("C_28","P_242"),("C_38","P_70"),("C_4","P_71")], "3V3加入C3下端/VOUT的真实同一节点")
        e.pin("C_38", "P_139", tip=[160,310], base=[160,311])
        e.pin("C_41", "P_140", tip=[160,310], base=[160,307])
        e.edge("E_120", [[[160,310],[160,310]]], [("C_38","P_139"),("C_41","P_140")], "恢复C3上端与上方GND的接触节点，不跨电容")
    elif e.case == "0102":
        e.edge("E_93", [[[1600,446],[1562,446]]], [("C_5","P_308")], "恢复QSPI_SS自身水平标号线")
        e.edge("E_101", [[[2090,475],[2091,475]]], [("C_5","P_312")], "恢复GPIO2原有短线，不跨接C34")
        e.component("C_28", [2003,280,2017,314], rotation=90)
        e.pin("C_28","P_121",tip=[2002,297],base=[2003,297])
        e.pin("C_28","P_122",tip=[2018,297],base=[2017,297])
        e.edge("E_31", [[[1963,257],[1971,257],[1971,303]], [[2002,297],[1971,297]]],
               [("C_14","P_76"),("C_5","P_77"),("C_28","P_121")], "C34左端接R8后的ADC_AVDD节点，补回原图可见支路")
        e.retire("E_186", "原GND符号误识别为电容，虚构的上端短线位于空白")
        e.retire("E_187", "GND符号与芯片边框误当电容/引脚，原竖线不代表该连接")
        e.component("C_48", [2069,277,2089,317], category="Gnd", categoryId="2", rotation=90)
        e.pin("C_48","P_133",state="deleted",edgeId=None,
              manual_relation_review=e.note("GND符号不存在第二个电容端子"))
        e.pin("C_5","P_135",state="deleted",edgeId=None,
              manual_relation_review=e.note("芯片右上角边框不是真实引脚"))
        e.pin("C_48","P_134",tip=[2068,297],base=[2069,297])
        e.edge(None, [[[2018,297],[2068,297]]], [("C_28","P_122"),("C_48","P_134")], "C34右端接旁边GND符号")
    elif e.case == "0124":
        e.component("C_33", [1065,65,1084,101], rotation=0)
        e.pin("C_33","P_104",tip=[1075,64],base=[1075,65])
        e.edge("E_69", [[[1075,64],[1075,64]]], [("C_33","P_104"),("C_39","P_105")], "恢复C8上端与上方GND接触点，修正10px横向偏移")
        e.pin("C_1","P_229",manual_relation_review=e.note("PB9原图有未连接叉号，撤销假连接并保留引脚"))
        p = e.new_pin("C_33", [1075,102], [1075,101])
        e.pin("C_37","P_16",tip=[1080,105],base=[1081,105])
        e.edge("E_2", [[[1072,105],[1080,105]], [[1075,102],[1075,105]]],
               [("C_1","P_15"),("C_37","P_16"),("C_33",p)], "C8下端接VDD/3.3V，原标签y110偏离真实y105")
    elif e.case == "0150":
        e.edge("E_3", [[[127,96],[128,96],[128,97],[159,97]]], [("C_12","P_85")], "恢复CC1水平标号线")
        e.edge("E_22", [[[127,153],[128,153],[128,154],[159,154]]], [("C_12","P_76")], "恢复DP2水平标号线")
        e.component("C_79", [118,256,136,269], rotation=0)
        e.pin("C_79","P_207",tip=[127,255],base=[127,256])
        conns = [("C_79","P_207")]
        for y, name, number in [(200,"GND","B1A12"),(211,"SHELL","1"),(222,"SHELL","2"),(234,"SHELL","3"),(245,"SHELL","4")]:
            conns.append(("C_12", e.new_pin("C_12", [127,y], [113,y], name, number)))
        e.edge("E_38", [[[127,200],[127,255]]], conns, "恢复原图可见B1A12与四个SHELL接地的竖直干线")
    elif e.case == "0180":
        for pid,y in [("P_93",410),("P_94",420)]:
            e.pin("C_1",pid,tip=[540,y],base=[550,y],
                  manual_relation_review=e.note("原图该引脚没有外部可见导线，保留真实端点，不赋予虚构网络"))
        e.pin("C_1","P_79",tip=[540,260],base=[550,260])
        e.pin("C_13","P_47",tip=[504,170],base=[501,170])
        e.edge("E_3", [[[504,170],[540,170],[540,260]]], [("C_13","P_47"),("C_1","P_79")], "P1.0接左上原有线路，补齐末端缺失的18px")
        p = e.new_pin("C_1", [700,260], [688,260], "VCC", "40")
        e.edge(None, [[[700,241],[700,260]]], [("C_34","P_76"),("C_1",p)], "VCC接右侧40脚，原图路径清晰")
    elif e.case == "0238":
        e.component("C_11", [350,170,387,190], rotation=0)
        e.pin("C_11","P_56",tip=[368,169],base=[368,170])
        e.edge("E_28", [[[368,154],[368,169]]], [("C_13","P_55"),("C_11","P_56")], "C3下端独立接地，不连接DREG")
        e.component("C_15", rotation=0)
        e.pin("C_15","P_59",tip=[145,294],base=[145,295])
        e.pin("C_15","P_60",tip=[145,334],base=[145,333])
        e.retire("E_30", "竖直电容被当作水平电容生成的右侧假短线")
        e.retire("E_31", "竖直电容被当作水平电容生成的左侧假短线")
        e.edge("E_11", [[[175,292],[145,292],[145,294]]], [("C_1","P_5"),("C_15","P_59")], "DREG恢复到自身去耦电容上端")
        e.edge("E_41", [[[145,334],[145,336]]], [("C_15","P_60"),("C_25","P_75")], "DREG去耦电容下端接自己的GND，修正电容方向")


def changes(before, after, prefix=""):
    """JSON-pointer audit with old values supports exact rollback, including missing keys."""
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            pointer = prefix + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before: result.append(dict(path=pointer, op="add", after=after[key]))
            elif key not in after: result.append(dict(path=pointer, op="remove", before=before[key]))
            else: result.extend(changes(before[key], after[key], pointer))
        return result
    return [] if before == after else [dict(path=prefix, op="replace", before=before, after=after)]


def validate_editor(e, rgb, size):
    evidence = []
    for eid in e.routes:
        edge = e.data["edges"][eid]
        support = Image.new("1", size)
        draw = ImageDraw.Draw(support)
        for path in polylines(edge["keyPointPaths"]["wire"]): draw.line(path, fill=1, width=1)
        mask = np.asarray(support, dtype=bool)
        fraction = float((rgb[mask].min(axis=1) < 224).mean())
        assert fraction == 1., (e.case, eid, "route has blank pixels", fraction)
        points = {tuple(p) for p in edge["coordinates"]}
        for pid, connection in edge["connect"].items():
            cid = connection["cpntId"]
            a = e.data["pins"][pid]; b = e.data["components"][cid]["pins"][pid]
            assert a == b and a["edgeId"] == eid and a["tip"] == connection["location"]
            assert tuple(a["tip"]) in points, (e.case, eid, pid, "pin off reviewed route")
        evidence.append(dict(edge_id=eid, centerline_pixels=int(mask.sum()), visible_ink_fraction=fraction))
    for cid, pid in e.changed_pins:
        pin = e.data["pins"][pid]
        assert pin == e.data["components"][cid]["pins"][pid]
        if pin.get("state") == "deleted": continue
        x,y = pin["tip"]
        assert 0 <= x < size[0] and 0 <= y < size[1]
        # Existing unchanged endpoint conventions can be a pixel off visible leads.
        assert rgb[max(0,round(y)-1):round(y)+2,max(0,round(x)-1):round(x)+2].min() < 224
        eid = pin.get("edgeId")
        if eid is not None:
            assert e.data["edges"][eid]["state"] != "deleted"
            assert pid in e.data["edges"][eid]["connect"]
    return evidence


def main():
    rows = read_rows(BASE / "manifest.jsonl")
    raw = {r["case_id"]:r for r in read_rows(ROOT / "data/manifest.jsonl")}
    protected = {}
    for path in (BASE/"manifest.jsonl", BASE/"training_manifest.jsonl", BASE/"frozen_validation_manifest.jsonl", ROOT/"weights/unet_wire_pin_v1.pt"):
        protected[str(path)] = sha(path)
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/"review").mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],"axes.unicode_minus":False})
    records = []; page_checks = []; all_operations = []; total_added = total_removed = total_pin_diff = 0
    for row in rows:
        case = row["case_id"]
        if case not in REASONS: continue
        source_input = Path(row["source_annotation_path"])
        for p in (source_input, Path(raw[case]["raw_annotation_path"]), Path(row["wire_mask"]), Path(row["pin_heatmap"]), Path(row["valid_mask"])):
            protected[str(p)] = sha(p)
        original = json.loads(source_input.read_text(encoding="utf-8"))
        e = Editor(case, original); reannotate(e)
        rgb = np.asarray(Image.open(row["image"]).convert("RGB"))
        checks = validate_editor(e, rgb, (row["width"],row["height"]))
        wire,pin,tips,stats = annotation_targets(e.data,row["width"],row["height"],3,2)
        assert stats["missing_physical_paths"] == 0
        assert stats.get("ignored_source_associations",0) == 0
        before_wire = np.asarray(Image.open(row["wire_mask"]))
        before_pin = np.asarray(Image.open(row["pin_heatmap"]))
        after = np.asarray(wire)
        added = (after>0)&(before_wire==0); removed = (after==0)&(before_wire>0)
        pin_diff = np.asarray(pin) != before_pin
        total_added += int(added.sum()); total_removed += int(removed.sum()); total_pin_diff += int(pin_diff.sum())
        audit = changes(original,e.data)
        folder = OUT/case; folder.mkdir(exist_ok=True)
        source = folder/"source_annotation.json"; write_json(source,e.data)
        write_json(folder/"source_changes.json",audit)
        wire.save(folder/"wire_mask.png"); pin.save(folder/"pin_heatmap.png")
        write_json(folder/"pin_points.json",tips)
        Image.new("L",(row["width"],row["height"]),255).save(folder/"valid_mask.png")
        Image.fromarray(added.astype(np.uint8)*255).save(folder/"added_pixels.png")
        Image.fromarray(removed.astype(np.uint8)*255).save(folder/"removed_pixels.png")
        ignored_before = row["supervision_ignored_pixels"]
        row.update(source_annotation_path=str(source.resolve()),source_annotation_sha256=sha(source),
                   relation_repair_source_annotation_path=str(source_input),relation_repair_source_annotation_sha256=sha(source_input),
                   source_changes_path=str((folder/"source_changes.json").resolve()),
                   wire_mask=str((folder/"wire_mask.png").resolve()),pin_heatmap=str((folder/"pin_heatmap.png").resolve()),
                   valid_mask=str((folder/"valid_mask.png").resolve()),stats=stats,structurally_usable_draft=usable(stats),
                   label_status="draft_locally_corrected_source_relations_not_full_page_certified",
                   supervision_ignored_pixels=0,supervised_wire_positive_pixels=stats["wire_positive_pixels"],
                   uncertain_pin_targets=[],supervision_decisions=[],relation_repair_operations=e.operations,
                   resolved_graph_associations=list(REASONS[case]),graph_review_pending=False)
        write_json(folder/"metadata.json",row)
        all_operations.extend(dict(case_id=case,**operation) for operation in e.operations)
        for eid,reason in REASONS[case].items():
            records.append(dict(case_id=case,edge_id=eid,action="retired_false_source_association_and_reannotated_visible_connections",
                                reason=reason,resolution="local_image_review",graph_relation_review_pending=False,
                                original_connect=original["edges"][eid]["connect"],
                                replacement_operations=e.operations,source_changes=row["source_changes_path"],full_page_certified=False))
        page_checks.append(dict(case_id=case,visible_routes=checks,retired_edges=e.retired,new_pins=e.new_pins,
                                changed_pins=[list(p) for p in sorted(e.changed_pins)],
                                restored_supervised_pixels=ignored_before,added_wire_pixels=int(added.sum()),removed_wire_pixels=int(removed.sum()),
                                pin_heatmap_changed_pixels=int(pin_diff.sum()),structurally_usable_draft=row["structurally_usable_draft"]))
        for index,roi in enumerate(ROIS[case],1):
            x0,y0,x1,y1 = roi; extent = [x0-.5,x1-.5,y1-.5,y0-.5]
            fig,axes = plt.subplots(1,3,figsize=(13,4.2),dpi=145,layout="constrained")
            axes[0].imshow(rgb[y0:y1,x0:x1],extent=extent,interpolation="nearest")
            axes[0].set_title("原图 / 蓝点=修订后端点")
            for cid,pid in sorted(e.changed_pins):
                p = e.data["pins"][pid]
                if p.get("state") == "deleted": continue
                x,y = p["tip"]
                if x0<=x<x1 and y0<=y<y1: axes[0].scatter(x,y,s=28,facecolors="none",edgecolors="#126de0",linewidths=1)
            for ax,values,title in [(axes[1],before_wire,"v5导线标签"),(axes[2],after,"v6纠正后的导线标签")]:
                ax.imshow(values[y0:y1,x0:x1],extent=extent,cmap="gray",vmin=0,vmax=255,interpolation="nearest"); ax.set_title(title)
            for ax in axes: ax.set_xlabel("原图x"); ax.set_ylabel("原图y"); ax.set_xlim(x0-.5,x1-.5); ax.set_ylim(y1-.5,y0-.5)
            fig.suptitle(f"{case} · 撤销错误关联，按可见线路重标；本页仍非完整认证")
            fig.savefig(REPORT/"review"/f"corrected_{case}_{index}.png",facecolor="white"); plt.close(fig)
        print(json.dumps(dict(case_id=case,corrected_false_associations=len(REASONS[case]),added=int(added.sum()),removed=int(removed.sum()))),flush=True)
    frozen = read_rows(BASE/"frozen_validation_manifest.jsonl")
    train = [r for r in rows if r["split"]=="train" and r["structurally_usable_draft"]]
    for name,items in [("manifest",rows),("training_manifest",train+frozen),("frozen_validation_manifest",frozen)]:
        (OUT/f"{name}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in items),encoding="utf-8")
    write_json(OUT/"graph_review_queue.json",[])
    for p in (OUT/"relation_repairs.json",REPORT/"relation_repairs.json"): write_json(p,records)
    write_json(REPORT/"pages.json",page_checks)
    write_json(REPORT/"operations.json",all_operations)
    assert len(records)==10 and len(page_checks)==6
    assert all(sha(p)==digest for p,digest in protected.items())
    summary = dict(status="ten_false_associations_locally_corrected_with_visible_routes",
                   corrected_source_associations=10,corrected_pages=6,pending_relations_in_this_queue=0,
                   actions=dict(Counter(op["action"] for op in all_operations)),
                   new_pins=sum(len(p["new_pins"]) for p in page_checks),
                   restored_supervised_pixels=sum(p["restored_supervised_pixels"] for p in page_checks),
                   remaining_ignored_pixels=sum(r.get("supervision_ignored_pixels",0) for r in rows),
                   added_wire_positive_pixels=total_added,removed_false_wire_positive_pixels=total_removed,
                   changed_pin_heatmap_pixels=total_pin_diff,train_candidates=len(train),frozen_validation_cases=len(frozen),
                   training_manifest_rows=len(train)+len(frozen),supervised_missing_paths=sum(r["stats"]["missing_physical_paths"] for r in rows),
                   other_structural_issues={r["case_id"]:{k:r["stats"][k] for k in ("unowned_root_pins","out_of_bounds_paths","invalid_pin_tips") if r["stats"][k]} for r in rows if r["case_id"] in REASONS and not r["structurally_usable_draft"]},
                   full_pages_certified=0,independent_human_review=False,training_started=False,default_weights_changed=False,
                   manifest=str((OUT/"manifest.jsonl").resolve()),training_manifest=str((OUT/"training_manifest.jsonl").resolve()),
                   manifest_sha256=sha(OUT/"manifest.jsonl"),training_manifest_sha256=sha(OUT/"training_manifest.jsonl"),
                   protected_inputs_sha256=protected,script_sha256=sha(Path(__file__)))
    for p in (OUT/"summary.json",REPORT/"summary.json"): write_json(p,summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="protected_inputs_sha256"},ensure_ascii=False),flush=True)


if __name__ == "__main__": main()
