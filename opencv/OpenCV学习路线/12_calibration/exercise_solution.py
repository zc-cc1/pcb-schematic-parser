from demo import calibrate,report

def main():
    rows=[]
    for noise in [0.0,.2,1.0]:
        metrics,_=calibrate(noise)
        rows.append(metrics)
    report(rows,'exercise_metrics.json')

if __name__ == '__main__':
    main()
