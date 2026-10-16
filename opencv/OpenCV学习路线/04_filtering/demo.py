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


def make_clean():
    img = np.full((220, 320), 70, np.uint8)
    cv2.rectangle(img, (30,30), (140,180), 160, -1)
    cv2.circle(img, (235,110), 60, 210, -1)
    cv2.putText(img, 'CV', (45,120), cv2.FONT_HERSHEY_SIMPLEX, 1, 240, 2)
    return img

def mse(a,b):
    # 转成浮点数之后再相减、平方，避免uint8溢出。
    return float(np.mean((a.astype(np.float32)-b.astype(np.float32))**2))

def main():
    clean = make_clean()
    rng = np.random.default_rng(42)
    noisy = np.clip(clean.astype(float)+rng.normal(0,18,clean.shape),0,255).astype(np.uint8)
    salt = clean.copy()
    chance = rng.random(clean.shape)
    salt[chance < .025] = 0
    salt[chance > .975] = 255
    gaussian = cv2.GaussianBlur(noisy, (5,5), 1.2)
    mean_filtered = cv2.blur(noisy, (5,5))
    median = cv2.medianBlur(salt, 3)
    low = (clean.astype(np.float32)*.25+85).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(low)
    # 直方图统计每个灰度出现多少次，再画成柱状图；左边对应0，右边对应255。
    histogram = cv2.calcHist([low], [0], None, [256], [0,256]).ravel()
    histogram_image = np.full((200,256,3),255,np.uint8)
    for level,count in enumerate(histogram):
        height = int(180*count/max(float(histogram.max()),1.0))
        cv2.line(histogram_image,(level,199),(level,199-height),(30,100,200),1)
    save('08_mean_blur.png',mean_filtered)
    save('09_low_contrast_histogram.png',histogram_image)
    for name,img in [('01_clean',clean),('02_gaussian_noise',noisy),('03_gaussian_blur',gaussian),
                     ('04_salt_pepper',salt),('05_median',median),('06_low_contrast',low),('07_clahe',clahe)]:
        save(name+'.png',img)
    report({'gaussian_noise_mse': mse(clean,noisy), 'gaussian_filtered_mse': mse(clean,gaussian),
            'salt_noise_mse': mse(clean,salt), 'median_filtered_mse': mse(clean,median)})

if __name__ == '__main__':
    main()
