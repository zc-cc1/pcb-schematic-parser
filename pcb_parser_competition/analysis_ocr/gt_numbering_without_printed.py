"""On pages with no printed designators at all, what numbering does the GT use?

If the published answer numbers passives 1..n per prefix, the only unknown is the
*order*; measuring how well reading order predicts it tells us how many points a
generated designator could earn on such pages.
"""
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DATASET = Path(r"C:\Users\Pc\Desktop\新建文件夹\Dataset300")
PUBLIC = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
REF = re.compile(r"^(?:IC|SW|TP|FB|FET|LED|JP|[RCLDQUXYJPFS])\d+$", re.I)
SPLIT = re.compile(r"^([A-Za-z]+)(\d+)$")


def printed_ref_count(case):
    std = next((DATASET / case).glob("*_CVJsonStd.json"), None)
    if std is None:
        return None
    doc = json.loads(std.read_text(encoding="utf-8"))
    return sum(1 for item in (doc.get("texts") or {}).values()
               if REF.match(str(item.get("content") or "").strip()))


def main():
    rows = []
    for folder in sorted(PUBLIC.iterdir()):
        if not folder.is_dir():
            continue
        case = folder.name
        count = printed_ref_count(case)
        if count is None or count > 0:
            continue                      # only pages with zero printed references
        f = next(folder.glob("*_target_named.json"), None)
        if f is None:
            continue
        gt = json.loads(f.read_text(encoding="utf-8"))
        height = max((v["bbox"][3] for v in gt["components"].values()), default=0)
        comps = [(k, v) for k, v in gt["components"].items()
                 if str(v.get("type")) in {"r", "c", "l", "d", "led"} and SPLIT.match(k)]
        if len(comps) < 4:
            continue
        # reading order: top-to-bottom then left-to-right (bottom-left origin -> -y)
        order = sorted(comps, key=lambda kv: (-kv[1]["bbox"][3], kv[1]["bbox"][0]))
        by_prefix = defaultdict(list)
        for key, _ in order:
            prefix, number = SPLIT.match(key).groups()
            by_prefix[prefix.upper()].append(int(number))
        consecutive = total_groups = 0
        mono = pairs = 0
        for prefix, numbers in by_prefix.items():
            if len(numbers) < 2:
                continue
            total_groups += 1
            if sorted(numbers) == list(range(1, len(numbers) + 1)):
                consecutive += 1
            for first, second in zip(numbers, numbers[1:]):
                pairs += 1
                mono += second > first
        rows.append({"case": case, "passives": len(comps), "groups": total_groups,
                     "consecutive_groups": consecutive, "pairs": pairs, "mono": mono})

    print(f"pages with zero printed references and >=4 GT passives: {len(rows)}")
    total_pairs = sum(r["pairs"] for r in rows)
    total_mono = sum(r["mono"] for r in rows)
    total_groups = sum(r["groups"] for r in rows)
    total_consec = sum(r["consecutive_groups"] for r in rows)
    print(f"  GT numbers are exactly 1..n in {total_consec}/{total_groups} prefix groups "
          f"({total_consec/max(1,total_groups):.1%})")
    print(f"  reading order increases the number in {total_mono}/{total_pairs} adjacent pairs "
          f"({total_mono/max(1,total_pairs):.1%})")
    med = statistics.median([r["mono"] / r["pairs"] for r in rows if r["pairs"]])
    print(f"  per-page median monotonicity: {med:.1%}")
    print("\n  pages where reading order predicts the number in >60% of pairs:")
    for row in sorted(rows, key=lambda r: -(r["mono"] / max(1, r["pairs"]))):
        ratio = row["mono"] / max(1, row["pairs"])
        if ratio > .6:
            print(f"    {row['case']}: {row['passives']} passives "
                  f"{row['consecutive_groups']}/{row['groups']} groups consecutive, "
                  f"mono {ratio:.0%}")


if __name__ == "__main__":
    sys.exit(main())
