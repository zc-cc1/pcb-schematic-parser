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

def main():
    parser=argparse.ArgumentParser(description='使用自己的棋盘照片标定普通针孔相机；不适用于鱼眼模型。')
    parser.add_argument('folder',type=Path)
    parser.add_argument('--cols',type=int,default=9,help='每行内角点数，不是格子数')
    parser.add_argument('--rows',type=int,default=6)
    parser.add_argument('--square',type=float,default=.025,help='每格实际边长，单位米')
    args=parser.parse_args()
    if args.cols<2 or args.rows<2 or args.square<=0:
        parser.error('内角点行列至少为2，格子尺寸必须大于0')
    obj=np.zeros((args.cols*args.rows,3),np.float32)
    obj[:,:2]=np.mgrid[0:args.cols,0:args.rows].T.reshape(-1,2)*args.square
    objects,images,used=[],[],[]
    size=None
    sample=None
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,40,.001)
    for path in sorted(args.folder.glob('*')):
        if path.suffix.lower() not in ['.png','.jpg','.jpeg','.bmp']:
            continue
        bgr=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
        if bgr is None:
            print(f'跳过不可解码文件: {path.name}')
            continue
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        current=(gray.shape[1],gray.shape[0])
        if size is not None and current!=size:
            raise ValueError('照片分辨率必须一致，请先整理照片')
        size=current
        found,corners=cv2.findChessboardCorners(gray,(args.cols,args.rows))
        if not found:
            print(f'未找到完整棋盘: {path.name}')
            continue
        corners=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objects.append(obj.copy())
        images.append(corners)
        used.append(path.name)
        sample=bgr
        drawn=bgr.copy()
        cv2.drawChessboardCorners(drawn,(args.cols,args.rows),corners,True)
        save(f'real_corners_{len(used):02}.png',drawn)
    if len(images)<10:
        raise ValueError(f'仅找到{len(images)}张有效照片，本课程至少收集10张，建议15至25张不同角度照片')
    rms,K,dist,rs,ts=cv2.calibrateCamera(objects,images,size,None,None)
    errors=[]
    for obj,img,r,t in zip(objects,images,rs,ts):
        predicted,_=cv2.projectPoints(obj,r,t,K,dist)
        errors.append(float(np.sqrt(np.mean(np.sum((img-predicted)**2,axis=2)))))
    save('real_undistorted.png',cv2.undistort(sample,K,dist))
    np.savez(OUT/'real_calibration.npz',K=K,dist=dist,image_size=np.array(size))
    report({'rms_pixels':float(rms),'K':K.tolist(),'dist':dist.ravel().tolist(),
            'per_image_rms':dict(zip(used,errors))},'real_metrics.json')

if __name__ == '__main__':
    main()
