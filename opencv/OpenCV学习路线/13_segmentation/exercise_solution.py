from demo import make_coins,segment,save,report

def main():
    image,mask=make_coins()
    rows=[]
    for ratio in [.3,.5,.65,.85]:
        _,seeds,_,colored,metrics=segment(image,mask,ratio)
        metrics['seed_ratio']=ratio
        rows.append(metrics)
        save(f'exercise_seeds_{ratio}.png',seeds)
        save(f'exercise_regions_{ratio}.png',colored)
    report(rows,'exercise_metrics.json')

if __name__ == '__main__':
    main()
