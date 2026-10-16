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


import argparse
import time

def make_frame(index):
    frame=np.full((240,320,3),35,np.uint8)
    x=30+index*3
    cv2.circle(frame,(x,120),20,(0,0,255),-1)
    cv2.putText(frame,f'frame {index}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1)
    return frame

def make_video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(320,240))
    if not writer.isOpened():
        raise RuntimeError('MJPG视频编码器不可用；请检查OpenCV视频后端。')
    try:
        for i in range(80):
            writer.write(make_frame(i))
    finally:
        writer.release()

def process(source, max_frames=300, show=False):
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'无法打开视频源: {source}')
    writer=None
    count=0
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:
        fps=20.0
    start=time.perf_counter()
    try:
        while count < max_frames:
            ok,frame=cap.read()
            if not ok:
                break
            h,w=frame.shape[:2]
            if writer is None:
                writer=cv2.VideoWriter(str(OUT/'02_annotated.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h))
                if not writer.isOpened():
                    raise RuntimeError('输出视频无法创建。')
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(hsv,(0,100,70),(10,255,255)) | cv2.inRange(hsv,(170,100,70),(179,255,255))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour)>100:
                    x,y,bw,bh=cv2.boundingRect(contour)
                    cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),2)
            writer.write(frame)
            if count in [0,40,79]:
                save(f'frame_{count:03}.png',frame)
            count+=1
            if show:
                cv2.imshow('video - Q to quit',frame)
                if cv2.waitKey(max(1,int(1000/fps))) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    if count==0:
        raise RuntimeError('视频已打开但未读到任何帧。')
    report({'processed_frames':count,'source_fps':fps,
            'loop_fps_including_io_and_optional_wait':count/max(time.perf_counter()-start,1e-9)})

def main():
    parser=argparse.ArgumentParser(description='默认生成并处理视频；--camera 0 可选打开摄像头。')
    parser.add_argument('--camera',type=int)
    parser.add_argument('--show',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.camera is None:
        path=OUT/'01_input.avi'
        make_video(path)
        source=str(path)
    else:
        source=args.camera
    process(source,show=args.show)

if __name__ == '__main__':
    main()
