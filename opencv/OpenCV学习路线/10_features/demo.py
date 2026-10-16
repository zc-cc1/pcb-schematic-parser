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


def make_pair():
    rng=np.random.default_rng(42)
    image=np.full((320,440),230,np.uint8)
    for _ in range(150):
        x,y=map(int,rng.integers([15,15],[425,305]))
        radius=int(rng.integers(2,8))
        cv2.circle(image,(x,y),radius,int(rng.integers(10,180)),-1)
    cv2.putText(image,'OpenCV 2026',(55,170),cv2.FONT_HERSHEY_SIMPLEX,1.1,0,3)
    src=np.float32([[0,0],[439,0],[439,319],[0,319]])
    dst=np.float32([[25,20],[410,10],[428,295],[10,308]])
    true_H=cv2.getPerspectiveTransform(src,dst)
    moved=cv2.warpPerspective(image,true_H,(440,320),borderValue=255)
    return image,moved,true_H

def align(first,second,ratio=.75):
    orb=cv2.ORB_create(nfeatures=1800)
    k1,d1=orb.detectAndCompute(first,None)
    k2,d2=orb.detectAndCompute(second,None)
    if d1 is None or d2 is None or len(d2)<2:
        raise ValueError('纹理不足，无法匹配描述子')
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs=matcher.knnMatch(d1,d2,k=2)
    good=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<ratio*pair[1].distance]
    if len(good)<4:
        raise ValueError(f'有效匹配只有{len(good)}个，至少需要4对非退化点')
    p1=np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    p2=np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H,inliers=cv2.findHomography(p1,p2,cv2.RANSAC,3.0)
    if H is None or inliers is None or int(inliers.sum())<4:
        raise ValueError('无法估计可信的单应矩阵')
    drawn=cv2.drawMatches(first,k1,second,k2,good,None,matchesMask=inliers.ravel().tolist(),
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return H,drawn,{'keypoints_first':len(k1),'keypoints_second':len(k2),
                    'good_matches':len(good),'inliers':int(inliers.sum())}

def main():
    first,second,true_H=make_pair()
    H,matches,metrics=align(first,second)
    restored=cv2.warpPerspective(second,np.linalg.inv(H),(first.shape[1],first.shape[0]))
    corners=np.float32([[0,0],[439,0],[439,319],[0,319]]).reshape(-1,1,2)
    actual=cv2.perspectiveTransform(corners,true_H)
    predicted=cv2.perspectiveTransform(corners,H)
    metrics['mean_corner_error_pixels']=float(np.linalg.norm(actual-predicted,axis=2).mean())
    for name,img in [('01_reference',first),('02_warped',second),('03_inlier_matches',matches),('04_aligned',restored)]:
        save(name+'.png',img)
    report(metrics)

if __name__ == '__main__':
    main()
