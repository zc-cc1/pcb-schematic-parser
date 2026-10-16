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

def make_parts(defect=True,background=30):
    image=np.full((280,480,3),background,np.uint8)
    for center in [(90,90),(240,90),(390,90),(90,210),(240,210),(390,210)]:
        cv2.circle(image,center,35,(220,220,220),-1)
    if defect:
        cv2.rectangle(image,(407,67),(432,108),(background,)*3,-1)  # 右上零件缺口
    return image

def inspect(image,threshold=110,min_area=400,min_circularity=.84):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,mask=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result=image.copy()
    objects=[]
    for contour in sorted(contours,key=lambda c:(cv2.boundingRect(c)[1],cv2.boundingRect(c)[0])):
        area=cv2.contourArea(contour)
        if area<min_area:
            continue
        perimeter=cv2.arcLength(contour,True)
        circularity=4*np.pi*area/(perimeter**2) if perimeter else 0
        defective=circularity<min_circularity
        x,y,w,h=cv2.boundingRect(contour)
        color=(0,0,255) if defective else (0,200,0)
        cv2.rectangle(result,(x,y),(x+w,y+h),color,2)
        cv2.putText(result,'NG' if defective else 'OK',(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        objects.append({'area':float(area),'circularity':float(circularity),'defective':bool(defective),'box':[x,y,w,h]})
    return mask,result,{'count':len(objects),'defects':sum(obj['defective'] for obj in objects),'objects':objects}

def main():
    parser=argparse.ArgumentParser(description='亮色、互不接触的圆形零件检测教学样例。')
    parser.add_argument('--input',type=Path)
    parser.add_argument('--threshold',type=int,default=110)
    parser.add_argument('--min-area',type=float,default=400)
    parser.add_argument('--min-circularity',type=float,default=.84)
    args=parser.parse_args()
    if not 0<=args.threshold<=255 or args.min_area<=0 or not 0<args.min_circularity<=1:
        parser.error('阈值需在0~255，面积需大于0，圆度门槛需在0~1之间')
    if args.input:
        image=cv2.imdecode(np.fromfile(str(args.input),np.uint8),cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('无法解码输入图片')
    else:
        image=make_parts()
    mask,result,metrics=inspect(image,args.threshold,args.min_area,args.min_circularity)
    save('01_input.png',image)
    save('02_mask.png',mask)
    save('03_inspected.png',result)
    metrics['configuration']={'threshold':args.threshold,'min_area':args.min_area,'min_circularity':args.min_circularity}
    report(metrics)

if __name__ == '__main__':
    main()
