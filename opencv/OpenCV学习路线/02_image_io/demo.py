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


def read_image(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"没有找到图片: {path}")
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"文件存在，但不是可解码的图片: {path}")
    return image

def make_scene():
    image = np.full((240, 360, 3), 70, np.uint8)
    cv2.rectangle(image, (30, 40), (160, 200), (255, 80, 20), -1)
    cv2.circle(image, (260, 120), 60, (20, 40, 240), -1)
    return image

def main():
    save('01_input.png', make_scene())
    image = read_image(OUT / '01_input.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (260, 120), 60, 255, -1)
    selected = cv2.bitwise_and(image, image, mask=mask)
    # 灰度先复制到3个通道，才可与彩色图拼接、按掩膜替换。
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    spotlight = gray_bgr.copy()
    spotlight[mask > 0] = image[mask > 0]
    save('02_gray.png', gray)
    save('03_mask.png', mask)
    save('04_selected.png', selected)
    save('05_spotlight.png', spotlight)
    save('06_blend.png', cv2.addWeighted(image, 0.6, gray_bgr, 0.4, 0))
    a = np.array([[250]], np.uint8)
    b = np.array([[10]], np.uint8)
    report({'gray_shape': list(gray.shape), 'numpy_add': int((a+b)[0, 0]),
            'opencv_add': int(cv2.add(a, b)[0, 0])})

if __name__ == '__main__':
    main()
