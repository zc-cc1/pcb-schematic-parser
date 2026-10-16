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


def make_document():
    # 横向渐变模拟不均匀照明；字比纸张暗。
    paper = np.tile(np.linspace(75,235,400).astype(np.uint8), (240,1))
    for y in [60,120,180]:
        cv2.putText(paper, 'OpenCV 123', (15,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 25, 2)
    return paper

def main():
    gray = make_document()
    _, fixed = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    threshold, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,31,9)
    binary = np.zeros((200,300),np.uint8)
    cv2.rectangle(binary,(35,35),(120,160),255,-1)
    cv2.rectangle(binary,(170,35),(260,160),255,-1)
    cv2.circle(binary,(210,85),7,0,-1)  # 内部黑洞
    rng=np.random.default_rng(42)
    for x,y in rng.integers([0,0],[300,200],size=(70,2)):
        binary[y,x]=255  # 散落白噪声
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opened=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
    closed=cv2.morphologyEx(opened,cv2.MORPH_CLOSE,np.ones((17,17),np.uint8))
    save('08_eroded.png',cv2.erode(binary,np.ones((7,7),np.uint8)))
    save('09_dilated.png',cv2.dilate(binary,np.ones((7,7),np.uint8)))
    for name,img in [('01_document',gray),('02_fixed',fixed),('03_otsu',otsu),
                     ('04_adaptive',adaptive),('05_dirty_mask',binary),('06_opened',opened),('07_closed',closed)]:
        save(name+'.png',img)
    report({'otsu_threshold': float(threshold), 'foreground_pixels_before': int(np.count_nonzero(binary)),
            'foreground_pixels_after': int(np.count_nonzero(closed))})

if __name__ == '__main__':
    main()
