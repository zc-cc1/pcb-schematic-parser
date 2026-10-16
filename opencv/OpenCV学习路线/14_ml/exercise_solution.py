from demo import train_model,evaluate,report,save,np

def main():
    svm=train_model()
    rows=[]
    for angle in [0,20,45]:
        metrics,pictures,_=evaluate(svm,angle)
        rows.append(metrics)
        save(f'exercise_rotated_{angle}.png',np.hstack(pictures[:10]))
    report(rows,'exercise_metrics.json')

if __name__ == '__main__':
    main()
