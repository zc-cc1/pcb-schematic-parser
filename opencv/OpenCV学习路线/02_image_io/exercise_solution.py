from demo import np, cv2, save, make_scene

def main():
    image = make_scene()
    roi = image[40:200, 30:160].copy()
    tiny = cv2.resize(roi, (8, 10), interpolation=cv2.INTER_AREA)
    image[40:200, 30:160] = cv2.resize(tiny, (130, 160), interpolation=cv2.INTER_NEAREST)
    # 在原本均匀的矩形内先放纹理，才能看出马赛克；重新演示一个有字的ROI。
    cv2.putText(image, 'PRIVATE', (35, 110), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    save('exercise_before.png', image)
    roi = image[40:200, 30:160]
    tiny = cv2.resize(roi, (8, 10), interpolation=cv2.INTER_AREA)
    image[40:200, 30:160] = cv2.resize(tiny, (130, 160), interpolation=cv2.INTER_NEAREST)
    save('exercise_mosaic.png', image)

if __name__ == '__main__':
    main()
