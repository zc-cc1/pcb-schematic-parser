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


def make_shapes():
    img=np.zeros((280,450),np.uint8)
    cv2.circle(img,(80,135),50,255,-1)
    cv2.rectangle(img,(170,70),(255,195),255,-1)
    pts=np.array([[345,55],[295,200],[410,200]],np.int32)
    cv2.fillPoly(img,[pts],255)
    return img

def analyze(mask, min_area=300):
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    canvas=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    rows=[]
    for contour in sorted(contours,key=lambda c:cv2.boundingRect(c)[0]):
        area=cv2.contourArea(contour)
        if area < min_area:
            continue
        length=cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,.02*length,True)
        M=cv2.moments(contour)
        if M['m00']==0:
            continue
        cx,cy=int(M['m10']/M['m00']),int(M['m01']/M['m00'])
        x,y,w,h=cv2.boundingRect(contour)
        circle_score=4*np.pi*area/(length*length) if length else 0
        cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,180,255),2)
        cv2.circle(canvas,(cx,cy),4,(0,0,255),-1)
        rows.append({'area':float(area),'vertices':len(approx),'centroid':[cx,cy],
                     'circularity':float(circle_score),'box':[x,y,w,h]})
    return canvas, rows

def main():
    mask=make_shapes()
    gx=cv2.Sobel(mask,cv2.CV_32F,1,0,ksize=3)
    edges=cv2.Canny(mask,60,150)
    result,rows=analyze(mask)
    save('01_shapes.png',mask)
    save('02_sobel_x.png',cv2.convertScaleAbs(gx))
    save('03_canny.png',edges)
    save('04_measurements.png',result)
    report({'count':len(rows),'objects':rows})

if __name__ == '__main__':
    main()
