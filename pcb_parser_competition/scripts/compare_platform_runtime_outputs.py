"""Compare deployment sources and smoke predictions across runtime images."""
import argparse
import json
from pathlib import Path


def differences(old, new, path=""):
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            current = path + "/" + key
            if key not in old or key not in new:
                yield current, old.get(key), new.get(key)
            else:
                yield from differences(old[key], new[key], current)
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for index, (left, right) in enumerate(zip(old, new)):
            yield from differences(left, right, path + "/" + str(index))
    elif old != new:
        yield path, old, new


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_root", type=Path)
    parser.add_argument("new_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_manifest = read(args.old_root / "source/package_manifest.json")
    new_manifest = read(args.new_root / "source/package_manifest.json")
    changed = [
        path for path in sorted(old_manifest["files"].keys() | new_manifest["files"].keys())
        if old_manifest["files"].get(path, {}).get("sha256")
        != new_manifest["files"].get(path, {}).get("sha256")
    ]
    assert set(changed) <= {"Dockerfile", "README.md"}, changed
    cases = {}
    for case in ("0036", "0042"):
        relative = Path("test-output/final_results") / (case + "_Pred.json")
        old, new = read(args.old_root / relative), read(args.new_root / relative)
        diff = list(differences(old, new))
        coordinates = []
        other = []
        for path, left, right in diff:
            numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                          for value in (left, right))
            is_coordinate = "/bbox/" in path or path.endswith(("/point/x", "/point/y"))
            if numeric and is_coordinate:
                coordinates.append((path, left, right))
            else:
                other.append((path, left, right))
        cases[case] = {
            "components": len(new["components"]),
            "pins": sum(len(pins) for pins in new["pins"].values()),
            "nets": len(new["nets"]),
            "nets_exactly_equal": old["nets"] == new["nets"],
            "coordinate_difference_count": len(coordinates),
            "max_coordinate_delta_px": max((abs(left-right) for _, left, right in coordinates), default=0),
            "other_differences": other,
            "coordinate_differences": coordinates,
        }
    report = {
        "old_root": str(args.old_root), "new_root": str(args.new_root),
        "changed_bundled_files": changed,
        "all_model_and_algorithm_files_identical": True,
        "old_smoke_device": read(args.old_root / "test-output/final_results/decision_log.json")["device"],
        "new_smoke_device": read(args.new_root / "test-output/final_results/decision_log.json")["device"],
        "cases": cases,
        "official_score": None,
        "limitation": "Two smoke cases only; this does not establish full-dataset or A800 GPU equivalence.",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"changed_bundled_files": changed, "cases": {
        case: {key: value for key, value in values.items() if key != "coordinate_differences"}
        for case, values in cases.items()}}, ensure_ascii=False, indent=2))
    assert all(not case["other_differences"] for case in cases.values()), "Non-coordinate outputs changed"


if __name__ == "__main__":
    main()
