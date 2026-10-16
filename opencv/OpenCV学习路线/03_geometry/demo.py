"""直接运行本文件；输入由代码生成，结果写入本周 outputs 文件夹。"""
from pathlib import Path
import json
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

def save(name, img):
    """用编码后写文件的方式兼容中文路径；保存失败立即给出错误。"""
    OUT.mkdir(exist_ok=True)
    ok, data = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"图片编码失败: {name}")
    data.tofile(str(OUT / name))

def report(data, name="metrics.json"):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def make_grid():
    img = np.full((240, 360, 3), 245, np.uint8)
    for x in range(0, 360, 30):
        cv2.line(img, (x, 0), (x, 239), (180, 180, 180), 1)
    for y in range(0, 240, 30):
        cv2.line(img, (0, y), (359, y), (180, 180, 180), 1)
    cv2.putText(img, 'OpenCV', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 60, 210), 3)
    return img

def main():
    img = make_grid()
    h, w = img.shape[:2]
    small = cv2.resize(img, (w//2, h//2), interpolation=cv2.INTER_AREA)
    M = cv2.getRotationMatrix2D((w/2, h/2), 25, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(30, 30, 30))
    src = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
    dst = np.float32([[55,20], [310,40], [345,215], [15,225]])
    H = cv2.getPerspectiveTransform(src, dst)
    tilted = cv2.warpPerspective(img, H, (w,h))
    restored = cv2.warpPerspective(tilted, np.linalg.inv(H), (w,h))
    save('01_grid.png', img)
    save('02_small.png', small)
    save('06_nearest_enlarged.png', cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST))
    save('07_linear_enlarged.png', cv2.resize(small, (w,h), interpolation=cv2.INTER_LINEAR))
    save('03_rotated_cropped.png', rotated)
    save('04_perspective.png', tilted)
    save('05_restored.png', restored)
    report({'input_h_w': [h,w], 'small_h_w': list(small.shape[:2]),
            'mean_absolute_restore_error': float(np.abs(img.astype(float)-restored).mean())})

if __name__ == '__main__':
    main()
