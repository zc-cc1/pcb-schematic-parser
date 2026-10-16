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


def brighten(img, value):
    # 先转换到能容纳大数的类型，计算后裁剪，再变回图片常用的 uint8。
    return np.clip(img.astype(np.int16) + value, 0, 255).astype(np.uint8)

def main():
    image = np.zeros((180, 300, 3), dtype=np.uint8)  # 高180、宽300、3个通道
    image[:] = (40, 40, 40)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
    for index, color in enumerate(colors):
        left = 20 + index * 90
        image[40:140, left:left + 70] = color
    crop = image[40:140, 20:90].copy()  # .copy() 让裁剪结果独立
    copied = image.copy()
    view = copied[0:20, 0:20]  # 不加copy的切片通常共享原数组
    view[:] = 255
    values = np.array([250], dtype=np.uint8)
    wrapped = values + np.array([10], dtype=np.uint8)
    save('01_colors.png', image)
    save('02_crop.png', crop)
    save('03_bright.png', brighten(image, 60))
    save('04_view_changed_parent.png', copied)
    report({'shape': list(image.shape), 'dtype': str(image.dtype),
            'pixel_y50_x30': image[50, 30].tolist(),
            'uint8_250_plus_10': int(wrapped[0]),
            'safe_250_plus_10': int(brighten(values, 10)[0])})

if __name__ == '__main__':
    main()
