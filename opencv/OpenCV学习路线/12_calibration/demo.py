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


def synthetic_observations(noise=0.0):
    # 棋盘内角点9列6行，格子边长0.025米；板面Z=0。
    obj=np.zeros((9*6,3),np.float32)
    obj[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)*.025
    K=np.array([[800.,0,320],[0,810.,240],[0,0,1.]])
    dist=np.array([-.12,.03,0.,0.,0.])
    rng=np.random.default_rng(42)
    objects,images=[],[]
    for index in range(18):
        rvec=rng.uniform(-.35,.35,3).astype(np.float64)
        tvec=np.array([rng.uniform(-.14,-.07),rng.uniform(-.10,-.04),rng.uniform(.65,.95)])
        projected,_=cv2.projectPoints(obj,rvec,tvec,K,dist)
        projected+=rng.normal(0,noise,projected.shape)
        objects.append(obj.copy())
        images.append(projected.astype(np.float32))
    return objects,images,K,dist

def calibrate(noise=0.0):
    objects,images,true_K,true_dist=synthetic_observations(noise)
    rms,K,dist,rvecs,tvecs=cv2.calibrateCamera(objects,images,(640,480),None,None)
    # 独立重新投影，验证报告的RMS定义。
    squared=[]
    for obj,observed,r,t in zip(objects,images,rvecs,tvecs):
        predicted,_=cv2.projectPoints(obj,r,t,K,dist)
        squared.extend(np.sum((observed-predicted)**2,axis=2).ravel().tolist())
    return {'rms_pixels':float(rms),'recomputed_rms_pixels':float(np.sqrt(np.mean(squared))),
            'estimated_K':K.tolist(),'true_K':true_K.tolist(),
            'estimated_distortion':dist.ravel().tolist(),'true_distortion':true_dist.tolist(),
            'noise_std_pixels':noise},images

def main():
    metrics,images=calibrate()
    board=np.full((480,640,3),245,np.uint8)
    cv2.drawChessboardCorners(board,(9,6),images[0],True)
    save('01_synthetic_corner_observations.png',board)
    report(metrics)
    # 这份结果只属于模拟相机，不能拿来校正自己的摄像头。
    OUT.mkdir(exist_ok=True)
    np.savez(OUT/'synthetic_calibration.npz',K=np.array(metrics['estimated_K']),
             dist=np.array(metrics['estimated_distortion']))

if __name__ == '__main__':
    main()
