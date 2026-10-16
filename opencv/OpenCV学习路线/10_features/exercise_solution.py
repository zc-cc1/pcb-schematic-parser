from demo import make_pair,align,report,save

def main():
    first,second,_=make_pair()
    rows=[]
    for ratio in [.55,.75,.95]:
        try:
            _,drawn,metrics=align(first,second,ratio)
            metrics['ratio']=ratio
            rows.append(metrics)
            save(f'exercise_ratio_{ratio}.png',drawn)
        except ValueError as error:
            rows.append({'ratio':ratio,'error':str(error)})
    report(rows,'exercise_metrics.json')

if __name__ == '__main__':
    main()
