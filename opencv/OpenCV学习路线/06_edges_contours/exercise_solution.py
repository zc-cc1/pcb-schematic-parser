from demo import np, cv2, save, report, make_shapes, analyze

def main():
    mask=make_shapes()
    canvas,rows=analyze(mask)
    for row in rows:
        n=row['vertices']
        label='triangle' if n==3 else ('quadrilateral' if n==4 else ('circle' if row['circularity']>.8 else 'other'))
        row['label']=label
        x,y,_,_=row['box']
        cv2.putText(canvas,label,(x,max(15,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,255,0),1)
    save('exercise_labels.png',canvas)
    report(rows,'exercise_metrics.json')

if __name__ == '__main__':
    main()
