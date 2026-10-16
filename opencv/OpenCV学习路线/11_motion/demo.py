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


def frame_at(x):
    frame=np.full((200,320,3),30,np.uint8)
    cv2.rectangle(frame,(x,70),(x+50,125),(220,220,220),-1)
    for i in range(3):
        cv2.circle(frame,(x+10+i*15,90),3,(0,0,0),-1)
    return frame

def main():
    first=frame_at(60)
    second=frame_at(72)
    a=cv2.cvtColor(first,cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(second,cv2.COLOR_BGR2GRAY)
    difference=cv2.absdiff(a,b)
    _,motion=cv2.threshold(difference,25,255,cv2.THRESH_BINARY)
    points=cv2.goodFeaturesToTrack(a,maxCorners=30,qualityLevel=.01,minDistance=5)
    if points is None:
        raise RuntimeError('没有找到可跟踪的角点')
    moved,status,error=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(21,21),maxLevel=3)
    if moved is None or status is None:
        raise RuntimeError('光流计算失败')
    valid=(status.ravel()==1) & np.isfinite(moved.reshape(-1,2)).all(axis=1)
    start=points.reshape(-1,2)[valid]
    end=moved.reshape(-1,2)[valid]
    if len(start)==0:
        raise RuntimeError('没有跟踪成功的点')
    canvas=second.copy()
    for p,q in zip(start,end):
        cv2.arrowedLine(canvas,tuple(np.rint(p).astype(int)),tuple(np.rint(q).astype(int)),(0,0,255),1)
    bg=cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=16,detectShadows=False)
    empty=np.full_like(first,30)
    for _ in range(30):
        bg.apply(empty,learningRate=.2)  # 先学习空背景
    foreground=bg.apply(second,learningRate=0)  # 此次演示冻结模型
    for name,img in [('01_first',first),('02_second',second),('03_frame_difference',motion),
                     ('04_optical_flow',canvas),('05_background_foreground',foreground)]:
        save(name+'.png',img)
    report({'tracked_points':len(start),'median_displacement':np.median(end-start,axis=0).tolist(),
            'expected_displacement':[12,0]})

if __name__ == '__main__':
    main()
