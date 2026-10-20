"""Draw what the OCR saw and what the pipeline decided each string was.

One picture, one page:
  * every OCR text box, coloured by the role the tokenizer gave it
  * the component boxes we exported, with their key
  * the pin tips we exported, with the pin number we attached

Nothing is inferred for the drawing: all three layers are the real pipeline
artefacts (`texts.json`, `pin_text_result.json`, `<case>_Pred.json`).
"""
import json
import sys
from pathlib import Path

BASE = Path(r"D:\pcb5_submission")
sys.path.insert(0, str(BASE / "src"))
from pcb_parser.component_text import tokens_from_texts  # noqa: E402

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

GT_ROOT = Path(r"C:\Users\Pc\Desktop\新建文件夹\赛题六公开数据集\赛题六公开数据集\200_train_cases")
OUT = Path(r"C:\Users\Pc\Desktop\新建文件夹\pcb_parser_competition\analysis_ocr\evidence")

COLOURS = {
    "designator": (230, 0, 0),        # red    - looked like a component reference
    "value": (0, 90, 230),            # blue   - looked like a rating / value
    "model": (0, 150, 0),             # green  - looked like a part number
    "label": (235, 130, 0),           # orange - looked like a net / pin name
    "number": (150, 0, 150),          # purple - a bare number
}
ROLE_TO_GROUP = {
    "designator": "designator",
    "capacitance": "value", "inductance": "value", "resistance": "value",
    "frequency": "value", "rating": "value",
    "model_candidate": "model",
    "net_or_pin_label": "label",
    "numeric": "number",
}


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "0002"
    work = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"D:\pcb_v6_check\work_0002")
    texts_path = next(work.glob("work/**/upstream/upstream/texts.json"))
    pin_path = next(work.glob("work/**/upstream/upstream/pin_text_result.json"))
    pred_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(
        r"D:\pcb_v6_check\out_v6_full\final_results") / f"{case}_Pred.json"
    image_path = next((GT_ROOT / case).glob("*.png"))

    texts = json.loads(texts_path.read_text(encoding="utf-8"))["texts"]
    tokens = tokens_from_texts(texts)
    pins = json.loads(pin_path.read_text(encoding="utf-8"))["pin_candidates"]
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    image = Image.open(image_path).convert("RGB")
    height = image.size[1]
    scale = 2 if image.width < 1400 else 1
    canvas = image.resize((image.width * scale, image.height * scale), Image.LANCZOS)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13 * scale)
    except OSError:
        font = ImageFont.load_default()

    counts = {key: 0 for key in COLOURS}
    for token in tokens:
        group = ROLE_TO_GROUP.get(token["role"])
        if group is None:
            continue
        counts[group] += 1
        box = [value * scale for value in token["bbox"]]
        draw.rectangle(box, outline=COLOURS[group], width=2)
        draw.text((box[0], max(0, box[1] - 14 * scale)), token["text"][:14], fill=COLOURS[group], font=font)

    # exported component boxes (bottom-left origin in the file)
    for key, comp in pred["components"].items():
        x1, y1, x2, y2 = comp["bbox"]
        box = [x1 * scale, (height - y2) * scale, x2 * scale, (height - y1) * scale]
        draw.rectangle(box, outline=(0, 160, 160), width=2)
        draw.text((box[0] + 2, box[3] + 1), key, fill=(0, 120, 120), font=font)

    # exported pins: the tip plus the number we attached
    for owner, group in (pred.get("pins") or {}).items():
        for pin_key, pin in group.items():
            x = float(pin["point"]["x"]) * scale
            y = (height - float(pin["point"]["y"])) * scale
            draw.ellipse([x - 3 * scale, y - 3 * scale, x + 3 * scale, y + 3 * scale], fill=(0, 0, 0))
            draw.text((x + 4 * scale, y - 6 * scale), pin_key[4:], fill=(0, 0, 0), font=font)

    legend = [
        ("designator / value / model / net label / bare number - what the OCR box looked like", (0, 0, 0)),
        (f"red {counts['designator']}  blue {counts['value']}  green {counts['model']}  "
         f"orange {counts['label']}  purple {counts['number']}   (grey boxes not drawn)", (0, 0, 0)),
        ("cyan box + label = exported component, black dot + digit = exported pin", (0, 0, 0)),
    ]
    y = canvas.height - 18 * scale * len(legend)
    draw.rectangle([0, y - 4, canvas.width, canvas.height], fill=(255, 255, 255))
    for text, colour in legend:
        draw.text((6, y), text, fill=colour, font=font)
        y += 18 * scale

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{case}_what_ocr_saw.png"
    canvas.save(target)
    print(f"{case}: {len(texts)} OCR rows -> {len(tokens)} tokens, exported "
          f"{len(pred['components'])} components / {sum(len(v) for v in pred['pins'].values())} pins")
    print("role counts drawn:", counts)
    print("->", target)


if __name__ == "__main__":
    sys.exit(main())
