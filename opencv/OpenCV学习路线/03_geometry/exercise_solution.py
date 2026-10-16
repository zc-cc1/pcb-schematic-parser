from demo import cv2, np, save, make_grid

def rotate_bound(img, angle):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw, nh = int(np.ceil(h*sin+w*cos)), int(np.ceil(h*cos+w*sin))
    M[0,2] += nw/2-w/2
    M[1,2] += nh/2-h/2
    return cv2.warpAffine(img, M, (nw,nh), borderValue=(30,30,30))

if __name__ == '__main__':
    save('exercise_rotate_bound.png', rotate_bound(make_grid(), 25))
