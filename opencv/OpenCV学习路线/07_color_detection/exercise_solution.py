from demo import np, cv2, save, report, make_scene, detect

def main():
    rows=[]
    base=make_scene()
    for factor in [.25,.5,1.0]:
        image=np.clip(base.astype(float)*factor,0,255).astype(np.uint8)
        mask,result,objects=detect(image)
        save(f'exercise_brightness_{factor}.png',result)
        rows.append({'brightness':factor,'detected':len(objects),'actual':3})
    report(rows,'exercise_metrics.json')

if __name__ == '__main__':
    main()
