from demo import np,cv2,save,report,make_template,match

def main():
    template=make_template()
    larger=cv2.resize(template,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_LINEAR)
    h,w=larger.shape
    scene=np.full((240,360),50,np.uint8)
    scene[80:80+h,150:150+w]=larger
    candidates=[]
    for scale in [.75,1,1.25,1.5,1.75]:
        resized=cv2.resize(template,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
        _,score,point=match(scene,resized)
        candidates.append({'scale':scale,'score':score,'point':list(point)})
    save('exercise_scaled_scene.png',scene)
    report({'candidates':candidates,'best':max(candidates,key=lambda x:x['score'])},'exercise_metrics.json')

if __name__ == '__main__':
    main()
