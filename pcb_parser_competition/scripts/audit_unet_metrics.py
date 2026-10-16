"""Same-prediction metric, alignment and duplicate/leakage sensitivity audit."""
from collections import Counter, defaultdict
import hashlib
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.fft import dctn
from skimage.morphology import skeletonize

from unet_audit_common import ROOT, OUT, read, write, image, truth, probability, metrics, counts, aggregate, val_cases, assert_protected
from pcb_parser.unet_data import tile_starts


def shift(mask, dx, dy):
    h, w = mask.shape; result = np.zeros_like(mask)
    x0, x1 = max(0, dx), min(w, w+dx); y0, y1 = max(0, dy), min(h, h+dy)
    result[y0:y1, x0:x1] = mask[y0-dy:y1-dy, x0-dx:x1-dx]
    return result


def relaxed(pred, gt, radius):
    p = int((pred & (ndimage.distance_transform_edt(~gt) <= radius)).sum())
    r = int((gt & (ndimage.distance_transform_edt(~pred) <= radius)).sum()) if pred.any() else 0
    return dict(pred_matched=p, gt_matched=r, pred_pixels=int(pred.sum()), gt_pixels=int(gt.sum()))


def relaxed_aggregate(records):
    a = {key: sum(r[key] for r in records) for key in ("pred_matched", "gt_matched", "pred_pixels", "gt_pixels")}
    p = a["pred_matched"]/a["pred_pixels"] if a["pred_pixels"] else 0
    r = a["gt_matched"]/a["gt_pixels"] if a["gt_pixels"] else 0
    return a | dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0)


def trim_identity(rgb):
    occupied = np.any(rgb < 250, axis=2)
    yy, xx = np.where(occupied)
    trimmed = rgb[yy.min():yy.max()+1, xx.min():xx.max()+1] if len(xx) else rgb
    hasher = hashlib.sha256(); hasher.update(str(trimmed.shape).encode()); hasher.update(trimmed.tobytes())
    return hasher.hexdigest()


def phash(rgb):
    small = np.asarray(Image.fromarray(rgb).convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
    values = dctn(small, norm="ortho")[:8, :8].ravel()
    return values > np.median(values[1:])


def main():
    assert_protected(); start = time.perf_counter()
    baseline = read(OUT/"baseline_reproduction.json")
    thresholds = read(OUT/"protocol.json")["thresholds"]
    offsets = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 2), (-1, 0), (0, -1), (-2, 0), (0, -2)]
    curve = defaultdict(list); boundaries = defaultdict(list); shifts = defaultdict(list); gt_alignment = defaultdict(list)
    skel_records = []; relaxed_records = defaultdict(list); cl_records = []
    tiles = []; cases = []
    for index, c in enumerate(val_cases()):
        rgb = image(c); gt = truth(c); p = probability(c); pred = p >= .5
        for threshold in thresholds:
            curve[str(threshold)].append(metrics(*counts(p >= threshold, gt)))
        for radius in (1, 2):
            band = ndimage.binary_dilation(gt, iterations=radius) & ~ndimage.binary_erosion(gt, iterations=radius)
            row = metrics(*counts(pred, gt, band)) | dict(region_pixels=int(band.sum()),
                error_pixels=int(((pred ^ gt) & band).sum()), GT_positive_pixels=int((gt & band).sum()))
            boundaries[str(radius)].append(row)
            relaxed_records[str(radius)].append(relaxed(pred, gt, radius))
        skgt = skeletonize(gt); skpred = skeletonize(pred)
        skel_records.append(metrics(*counts(skpred, skgt)))
        cl_records.append(dict(pred_matched=int((skpred & gt).sum()), gt_matched=int((skgt & pred).sum()),
                               pred_pixels=int(skpred.sum()), gt_pixels=int(skgt.sum())))
        for dx, dy in offsets:
            moved = shift(gt, dx, dy)
            shifts[str((dx, dy))].append(metrics(*counts(moved, gt)) | dict(lost_GT_at_canvas=int(gt.sum())-int(moved.sum())))
            gt_alignment[str((dx, dy))].append(metrics(*counts(pred, moved)))
        # Image/GT alignment cue: mean ink at translated GT skeleton locations.
        # It is not semantic truth; component/text overlap requires visual review.
        gray = .299*rgb[..., 0]+.587*rgb[..., 1]+.114*rgb[..., 2]
        ink = 255-gray
        correlations = []
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                mask = shift(skgt, dx, dy)
                correlations.append(dict(dx=dx, dy=dy, mean_ink=float(ink[mask].mean()) if mask.any() else 0))
        best = max(correlations, key=lambda r: r["mean_ink"])
        at_zero = next(r for r in correlations if r["dx"] == r["dy"] == 0)
        zero_gt = int((gt & (ink < .5)).sum())
        page = metrics(*counts(pred, gt))
        cases.append(dict(case_id=c, wire=page, alignment_peak=best, alignment_zero=at_zero, alignment_curve=correlations,
            GT_positive_pixels=int(gt.sum()), GT_on_pure_white_pixels=zero_gt,
            pure_white_fraction_of_GT=zero_gt/int(gt.sum()), skeleton_dice=skel_records[-1]["dice"],
            relaxed1=relaxed_records["1"][-1], relaxed2=relaxed_records["2"][-1]))
        h, w = gt.shape
        for y in tile_starts(h, 512, 384):
            for x in tile_starts(w, 512, 384):
                a, b = pred[y:y+512, x:x+512], gt[y:y+512, x:x+512]
                tiles.append(metrics(*counts(a, b), empty=1.) | dict(case_id=c, x=x, y=y, gt_positive=bool(b.any())))
        print(json.dumps(dict(case_id=c, completed=index+1, total=26, alignment_peak=best)), flush=True)
    def sum_metric(items):
        return metrics(*(sum(r[k] for r in items) for k in ("tp", "fp", "fn")))
    curves = [dict(threshold=float(t), micro=sum_metric(values), macro_dice=float(np.mean([r["dice"] for r in values])))
              for t, values in curve.items()]
    curves.sort(key=lambda r: r["threshold"])
    best_curve = max(curves, key=lambda r: r["micro"]["dice"])
    boundary_summary = {}
    errors = baseline["micro"]["fp"]+baseline["micro"]["fn"]
    for radius, values in boundaries.items():
        row = sum_metric(values)
        boundary_summary[radius] = row | dict(region_pixels=sum(r["region_pixels"] for r in values),
            error_pixels=sum(r["error_pixels"] for r in values),
            fraction_of_total_errors=sum(r["error_pixels"] for r in values)/errors,
            fraction_of_GT_inside_band=sum(r["GT_positive_pixels"] for r in values)/sum(c["GT_positive_pixels"] for c in cases))
    gt_shifts = {key: sum_metric(values) | dict(macro_dice=float(np.mean([r["dice"] for r in values])),
        lost_GT_at_canvas=sum(r["lost_GT_at_canvas"] for r in values)) for key, values in shifts.items()}
    diagnostic_gt_offsets = {key: sum_metric(values) for key, values in gt_alignment.items()}
    # Exact/cropped-identity and pHash checks cover the entire existing split.
    page_records = baseline["records"]; trimmed = defaultdict(list); hashes = {}
    for r in page_records:
        rgb = image(r["case_id"])
        trimmed[trim_identity(rgb)].append(dict(case_id=r["case_id"], split=r["split"]))
        hashes[r["case_id"]] = phash(rgb)
    trim_dups = [dict(cases=values, cross_split=len(set(r["split"] for r in values)) > 1) for values in trimmed.values() if len(values) > 1]
    train = [r["case_id"] for r in page_records if r["split"] == "train"]
    val = [r["case_id"] for r in page_records if r["split"] == "val"]
    near = sorted([dict(train=a, val=b, hamming=int((hashes[a] != hashes[b]).sum())) for a in train for b in val], key=lambda r:r["hamming"])
    candidates = [r for r in near if r["hamming"] <= 8]
    meta = dict(whole_image_grouped_evaluation_dice=baseline["micro"]["dice"],
        whole_image_grouped_delta=0., reason="No shared exact full-image group crosses split; case assignment precedes tiles",
        exact_cross_split_groups=baseline["cross_split_identical_groups"], trim_cross_split_groups=sum(r["cross_split"] for r in trim_dups),
        trimmed_groups=trim_dups, phash_candidates=candidates, phash_top20=near[:20], phash_is_not_proof=True,
        engineering_grouped_Dice=None, engineering_grouped_reason="300 source IDs null and user has no IDs; frozen checkpoint cannot undo training exposure by reshuffling evaluation pages")
    tp, fp, fn = (baseline["micro"][k] for k in ("tp", "fp", "fn"))
    result = dict(status="same_checkpoint_same_predictions_metric_audit_complete", baseline_micro=baseline["micro"],
        baseline_macro=float(np.mean([r["wire"]["dice"] for r in cases])), threshold_curve=curves,
        diagnostic_best_threshold=best_curve, diagnostic_best_threshold_not_adopted=True,
        skeleton_micro=sum_metric(skel_records), relaxed_micro={r:relaxed_aggregate(v) for r,v in relaxed_records.items()},
        topology_clDice=relaxed_aggregate(cl_records), boundary_band=boundary_summary,
        GT_shift_sensitivity=gt_shifts, prediction_vs_shifted_GT_diagnostic=diagnostic_gt_offsets,
        GT_shift_is_not_annotation_ceiling=True, inter_annotator_Dice=None,
        all_background_images=baseline["stats"]["val26_unique"]["all_background_images"],
        excluding_background_images_micro=baseline["micro"],
        overlapping_valid_tiles_from_same_stitched_predictions=dict(micro=sum_metric(tiles), macro_with_empty_perfect=float(np.mean([r["dice"] for r in tiles])),
            macro_without_GT_empty=float(np.mean([r["dice"] for r in tiles if r["gt_positive"]])), tiles=len(tiles), GT_empty=sum(not r["gt_positive"] for r in tiles),
            note="Same stitched probability field sliced into actual valid tile rectangles; overlap counted again; no image resizing"),
        alignment_peak_counts=dict(Counter(str((r["alignment_peak"]["dx"],r["alignment_peak"]["dy"])) for r in cases)),
        GT_pure_white_fraction=sum(r["GT_on_pure_white_pixels"] for r in cases)/sum(r["GT_positive_pixels"] for r in cases),
        leakage=meta, cases=cases, oracle_bounds=dict(fix_all_FP_only=metrics(tp,0,fn), fix_all_FN_only=metrics(tp+fn,fp,0),
            perfect_match_GTs_Dice=1., annotation_ceiling_inferred=False),
        protected_files=assert_protected(), elapsed_seconds=time.perf_counter()-start)
    write(OUT/"metric_audit.json", result)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160, layout="constrained")
    ax.plot([r["threshold"] for r in curves], [r["micro"]["dice"]*100 for r in curves], "o-", label="整图 micro")
    ax.plot([r["threshold"] for r in curves], [r["macro_dice"]*100 for r in curves], "s-", label="按图 macro")
    ax.axvline(.5, color="#888888", ls="--", label="当前固定阈值0.5")
    ax.set(xlabel="导线概率阈值（扫描仅诊断，不采用）", ylabel="Dice（%）", title="同一checkpoint与概率图：阈值敏感性")
    ax.grid(alpha=.2); ax.legend(); fig.savefig(OUT/"dice_threshold_curve.png", facecolor="white"); plt.close(fig)
    print(json.dumps(dict(baseline_micro=result["baseline_micro"], macro=result["baseline_macro"], best_threshold=best_curve,
        skeleton=result["skeleton_micro"], relaxed=result["relaxed_micro"], boundary=boundary_summary,
        gt_shifts=gt_shifts, empty_tiles=result["overlapping_valid_tiles_from_same_stitched_predictions"],
        alignment_peak_counts=result["alignment_peak_counts"], cross_split_trim=meta["trim_cross_split_groups"], phash_candidates=len(candidates))), flush=True)


if __name__ == "__main__":
    main()
