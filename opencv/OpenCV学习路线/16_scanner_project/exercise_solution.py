from demo import np,cv2,make_document,scan,save,report

def main():
    image=make_document()
    cases={'normal':image,'blank':np.full_like(image,35),'cropped':image[100:400,150:400]}
    rows=[]
    for name,case in cases.items():
        try:
            _,annotated,warped,_,_=scan(case)
            save(f'exercise_{name}_detected.png',annotated)
            save(f'exercise_{name}_rectified.png',warped)
            rows.append({'case':name,'detected':True,'expected_success':name=='normal'})
        except ValueError as error:
            rows.append({'case':name,'detected':False,'expected_success':name=='normal','reason':str(error)})
    report(rows,'exercise_metrics.json')

if __name__ == '__main__':
    main()
