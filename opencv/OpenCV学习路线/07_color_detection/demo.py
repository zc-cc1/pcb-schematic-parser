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


def red_mask(bgr, min_s=100, min_v=70):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    # uint8 HSV 的 H 范围是0~179，红色跨越首尾两段。
    low=cv2.inRange(hsv,(0,min_s,min_v),(10,255,255))
    high=cv2.inRange(hsv,(170,min_s,min_v),(179,255,255))
    return cv2.bitwise_or(low,high)

def make_scene():
    img=np.full((260,420,3),190,np.uint8)
    for center,color in [((70,80),(0,0,230)),((190,80),(40,0,230)),((330,80),(0,220,0)),
                         ((70,190),(220,0,0)),((210,190),(0,0,150))]:
        cv2.circle(img,center,32,color,-1)
    cv2.circle(img,(360,210),3,(0,0,255),-1)  # 应排除的小红点
    return img

def detect(img, min_area=200):
    mask=red_mask(img)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,connectivity=8)
    rows=[]
    canvas=img.copy()
    for label in range(1,n):  # 0号是背景
        x,y,w,h,area=map(int,stats[label])
        if area < min_area:
            continue
        rows.append({'area':area,'center':centers[label].tolist()})
        cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,255,255),2)
    return mask,canvas,rows

def main():
    image=make_scene()
    mask,result,rows=detect(image)
    save('01_scene.png',image)
    save('02_red_mask.png',mask)
    save('03_counted.png',result)
    report({'red_count':len(rows),'expected_count':3,'objects':rows})

if __name__ == '__main__':
    main()
