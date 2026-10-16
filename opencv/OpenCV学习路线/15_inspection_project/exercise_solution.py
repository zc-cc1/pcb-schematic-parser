from demo import make_parts,inspect,report,save

def main():
    rows=[]
    for name,defect,background in [('normal',False,30),('notch',True,30),('bright_background',True,150)]:
        image=make_parts(defect,background)
        _,result,metrics=inspect(image)
        expected_defects=int(defect)
        rows.append({'case':name,'expected_count':6,'expected_defects':expected_defects,
                     'actual_count':metrics['count'],'actual_defects':metrics['defects'],
                     'passed':metrics['count']==6 and metrics['defects']==expected_defects})
        save(f'exercise_{name}.png',result)
    # 最后一个用例故意违反暗背景假设；把失败记录下来才是完整实验。
    report(rows,'exercise_metrics.json')

if __name__ == '__main__':
    main()
