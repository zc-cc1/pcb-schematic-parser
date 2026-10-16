from demo import np, cv2, save, report, make_document

def main():
    gray=make_document()
    rows=[]
    for block in [11,31,71]:
        for c in [3,15]:
            mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV,block,c)
            save(f'exercise_block_{block}_c_{c}.png',mask)
            rows.append({'block':block,'C':c,'white_pixels':int(np.count_nonzero(mask))})
    report(rows,'exercise_metrics.json')

if __name__ == '__main__':
    main()
