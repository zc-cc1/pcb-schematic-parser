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


HOG=cv2.HOGDescriptor((32,32),(16,16),(8,8),(8,8),9)

def sample(rng,label,angle=0):
    image=np.zeros((32,32),np.uint8)
    cx,cy=map(int,rng.integers(14,19,2))
    radius=int(rng.integers(7,11))
    if label==0:
        cv2.circle(image,(cx,cy),radius,220,-1)
    else:
        cv2.rectangle(image,(cx-radius,cy-radius),(cx+radius,cy+radius),220,-1)
    if angle:
        M=cv2.getRotationMatrix2D((16,16),angle,1)
        image=cv2.warpAffine(image,M,(32,32))
    return np.clip(image.astype(float)+rng.normal(0,7,image.shape),0,255).astype(np.uint8)

def dataset(seed,count,angle=0):
    rng=np.random.default_rng(seed)
    labels=np.arange(count,dtype=np.int32)%2
    pictures=[sample(rng,int(label),angle) for label in labels]
    features=np.array([HOG.compute(img).ravel() for img in pictures],np.float32)
    return features,labels,pictures

def train_model():
    features,labels,_=dataset(42,240)
    svm=cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    if not svm.train(features,cv2.ml.ROW_SAMPLE,labels):
        raise RuntimeError('SVM训练失败')
    return svm

def evaluate(svm,angle=0):
    features,labels,pictures=dataset(2026,100,angle)
    _,predicted=svm.predict(features)
    predicted=predicted.ravel().astype(int)
    confusion=np.zeros((2,2),np.int32)
    for actual,pred in zip(labels,predicted):
        confusion[actual,pred]+=1
    return {'accuracy':float(np.mean(labels==predicted)),
            'confusion_rows_true_cols_predicted':confusion.tolist(),
            'feature_dimension':features.shape[1],'test_rotation_degrees':angle},pictures,predicted

def main():
    svm=train_model()
    metrics,pictures,predicted=evaluate(svm)
    montage=np.zeros((4*48,5*48,3),np.uint8)
    for i,img in enumerate(pictures[:20]):
        tile=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        row,col=divmod(i,5)
        montage[row*48:row*48+32,col*48:col*48+32]=tile
        cv2.putText(montage,str(predicted[i]),(col*48,row*48+44),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
    save('01_predictions_0_circle_1_square.png',montage)
    # FileStorage使用内存字符串保存，避免部分构建对中文模型路径的限制。
    storage=cv2.FileStorage('model.yml',cv2.FILE_STORAGE_WRITE|cv2.FILE_STORAGE_MEMORY)
    svm.write(storage)
    model_text=storage.releaseAndGetString()
    (OUT/'svm.yml').write_text(model_text,encoding='utf-8')
    report(metrics)

if __name__ == '__main__':
    main()
