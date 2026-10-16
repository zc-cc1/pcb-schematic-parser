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

def make_document():
    paper=np.full((420,300,3),245,np.uint8)
    cv2.putText(paper,'OpenCV NOTES',(22,50),cv2.FONT_HERSHEY_SIMPLEX,.7,(25,25,25),2)
    for y in range(90,380,28):
        cv2.line(paper,(25,y),(270,y),(60,60,60),2)
    src=np.float32([[0,0],[299,0],[299,419],[0,419]])
    dst=np.float32([[120,45],[435,85],[465,440],[65,420]])
    H=cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(paper,H,(560,500),borderValue=(35,35,35))

def order_points(points):
    points=np.asarray(points,dtype=np.float32).reshape(4,2)
    # 先围绕中心按角度排序，避免和/差取极值时重复选中同一个角点。
    center=points.mean(axis=0)
    angles=np.arctan2(points[:,1]-center[1],points[:,0]-center[0])
    ordered=points[np.argsort(angles)]
    start=int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered,-start,axis=0)  # 常规拍摄姿态下：左上、右上、右下、左下

def scan(image):
    h,w=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(5,5),0)
    edges=cv2.Canny(blurred,50,150)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidate=None
    for contour in sorted(contours,key=cv2.contourArea,reverse=True):
        area=cv2.contourArea(contour)
        if area < .15*h*w:
            continue
        approximate=cv2.approxPolyDP(contour,.02*cv2.arcLength(contour,True),True)
        if len(approximate)==4 and cv2.isContourConvex(approximate):
            candidate=order_points(approximate)
            break
    if candidate is None:
        raise ValueError('未找到足够大的完整四边形；请让纸张四角入镜、增强背景对比度')
    tl,tr,br,bl=candidate
    width=int(round(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl))))
    height=int(round(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr))))
    if min(width,height)<20:
        raise ValueError('检测到的区域过小')
    target=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    H=cv2.getPerspectiveTransform(candidate,target)
    warped=cv2.warpPerspective(image,H,(width,height))
    warped_gray=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
    binary=cv2.adaptiveThreshold(warped_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,12)
    annotated=image.copy()
    cv2.polylines(annotated,[candidate.astype(np.int32)],True,(0,255,0),3)
    return edges,annotated,warped,binary,candidate

def main():
    parser=argparse.ArgumentParser(description='默认扫描合成文档，--input 可传入自己的照片。')
    parser.add_argument('--input',type=Path)
    args=parser.parse_args()
    if args.input:
        image=cv2.imdecode(np.fromfile(str(args.input),np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('无法解码图片')
    else:
        image=make_document()
    save('01_input.png',image)
    try:
        edges,annotated,warped,binary,corners=scan(image)
    except ValueError as error:
        report({'success':False,'reason':str(error)})
        raise
    for name,img in [('02_edges',edges),('03_detected',annotated),('04_rectified',warped),('05_scan',binary)]:
        save(name+'.png',img)
    report({'success':True,'corners':corners.tolist(),'output_height_width':list(binary.shape)})

if __name__ == '__main__':
    main()
