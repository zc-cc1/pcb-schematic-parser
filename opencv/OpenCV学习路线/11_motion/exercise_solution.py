from demo import cv2,np,save,report,frame_at

def main():
    stationary=frame_at(60)
    brighter=cv2.convertScaleAbs(stationary,alpha=1,beta=35)
    a=cv2.cvtColor(stationary,cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(brighter,cv2.COLOR_BGR2GRAY)
    _,mask=cv2.threshold(cv2.absdiff(a,b),25,255,cv2.THRESH_BINARY)
    save('exercise_lighting_false_alarm.png',mask)
    report({'changed_pixel_fraction':float(np.count_nonzero(mask)/mask.size),
            'object_actually_moved':False},'exercise_metrics.json')

if __name__ == '__main__':
    main()
