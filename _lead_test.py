"""单测：引线检测 + 吸附，和 GT 逐点比。"""
import json
import math
from pathlib import Path

import pin_repair_library as R

GT = Path(r"C:/Users/Pc/Desktop/新建文件夹/赛题六公开数据集/赛题六公开数据集/open_GT/open_GT")

for case, component in [("1", "USB1"), ("13", "H1"), ("11", "USB2"), ("4", "U2")]:
    truth = json.loads((GT / case / f"{case}.json").read_text(encoding="utf-8"))
    image = R.read_image(next((GT / case).glob("*.png")))
    height = image.shape[0]
    bbox = truth["components"][component]["bbox"]
    px_box = [bbox[0], height - bbox[3], bbox[2], height - bbox[1]]
    leads = R.find_leads(image, px_box)
    print(f"=== case {case} {component}: {len(leads)} 条引线")

    # 用 GT 的引脚点模拟"已有引脚"，看吸附后离 GT 多远
    fake = {k: {"pinname": "", "point": {"x": v["point"]["x"], "y": v["point"]["y"]}}
            for k, v in truth["pins"][component].items()}
    before = {k: (v["point"]["x"], v["point"]["y"]) for k, v in fake.items()}
    moved = R.snap_pins(fake, leads, px_box, height)
    exact = 0
    for key, pin in fake.items():
        d = math.hypot(pin["point"]["x"] - before[key][0], pin["point"]["y"] - before[key][1])
        print(f"   {key:6s} GT=({before[key][0]:6.1f},{before[key][1]:6.1f}) "
              f"吸附后=({pin['point']['x']:6.1f},{pin['point']['y']:6.1f}) 移动 {d:5.1f}px")
    print(f"   移动了 {moved} 个")
