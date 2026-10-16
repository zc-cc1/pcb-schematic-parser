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


def make_template():
    template=np.full((42,54),180,np.uint8)
    cv2.rectangle(template,(2,2),(51,39),30,2)
    cv2.line(template,(8,30),(43,8),250,3)
    cv2.circle(template,(14,13),5,40,-1)
    return template

def match(image,template):
    if image.shape[0]<template.shape[0] or image.shape[1]<template.shape[1]:
        raise ValueError('模板不能大于待搜索图片')
    if float(template.std())<1e-6:
        raise ValueError('本例使用相关系数，不能用没有纹理的恒定模板')
    response=cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
    _,score,_,point=cv2.minMaxLoc(response)
    return response,float(score),point

def main():
    template=make_template()
    rng=np.random.default_rng(42)
    scene=rng.integers(20,80,(240,360),dtype=np.uint8)
    scene[90:132,170:224]=template
    response,score,(x,y)=match(scene,template)
    result=cv2.cvtColor(scene,cv2.COLOR_GRAY2BGR)
    h,w=template.shape
    cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,255),2)
    heatmap=cv2.applyColorMap(np.clip((response+1)*127.5,0,255).astype(np.uint8),cv2.COLORMAP_JET)
    save('01_scene.png',scene)
    save('02_template.png',template)
    save('03_response.png',heatmap)
    save('04_match.png',result)
    report({'best_top_left':[x,y],'expected_top_left':[170,90],
            'score':score,'accepted_at_0_8':score>=.8,'response_shape':list(response.shape)})

if __name__ == '__main__':
    main()
