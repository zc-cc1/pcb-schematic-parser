from demo import cv2, np, save, report, OUT, make_video

def main():
    OUT.mkdir(exist_ok=True)
    path=OUT/'exercise_source.avi'
    make_video(path)
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError('无法打开练习视频')
    index=0
    kept=[]
    try:
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            if index % 20==0:
                save(f'exercise_second_{index//20:02}.png',frame)
                kept.append(index)
            index+=1
    finally:
        cap.release()
    report({'saved_frame_indices':kept,'total_frames':index},'exercise_metrics.json')

if __name__ == '__main__':
    main()
