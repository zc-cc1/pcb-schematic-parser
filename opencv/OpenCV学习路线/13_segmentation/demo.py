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


def make_coins():
    mask=np.zeros((240,360),np.uint8)
    cv2.circle(mask,(130,120),62,255,-1)
    cv2.circle(mask,(228,120),62,255,-1)
    image=np.full((240,360,3),35,np.uint8)
    image[mask>0]=(200,200,200)
    return image,mask

def segment(image,mask,seed_ratio=.65):
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    _,sure_fg=cv2.threshold(distance,seed_ratio*distance.max(),255,cv2.THRESH_BINARY)
    sure_fg=sure_fg.astype(np.uint8)
    sure_bg=cv2.dilate(mask,np.ones((3,3),np.uint8),iterations=3)
    unknown=cv2.subtract(sure_bg,sure_fg)
    n,markers=cv2.connectedComponents(sure_fg)
    markers=markers.astype(np.int32)+1  # 背景1，物体从2开始
    markers[unknown>0]=0  # 0表示尚未决定属于谁
    cv2.watershed(image,markers)  # 原地修改markers，边界变成-1
    colored=image.copy()
    colors=[(50,160,240),(240,150,50),(60,220,80),(210,70,200)]
    labels=[int(v) for v in np.unique(markers) if v>1]
    for index,label in enumerate(labels):
        colored[markers==label]=colors[index%len(colors)]
    colored[markers==-1]=(0,0,255)
    return distance,sure_fg,markers,colored,{'seed_count':n-1,'regions':len(labels)}

def main():
    image,mask=make_coins()
    distance,seeds,markers,result,metrics=segment(image,mask)
    components,_=cv2.connectedComponents(mask)
    metrics['connected_objects_before']=components-1
    save('01_touching_coins.png',image)
    save('02_distance.png',cv2.normalize(distance,None,0,255,cv2.NORM_MINMAX).astype(np.uint8))
    save('03_seeds.png',seeds)
    save('04_separated.png',result)
    report(metrics)

if __name__ == '__main__':
    main()
