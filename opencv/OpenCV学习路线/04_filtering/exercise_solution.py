from demo import np, cv2, save, report, make_clean, mse

def main():
    clean = make_clean()
    rng = np.random.default_rng(7)
    noisy = np.clip(clean.astype(float)+rng.normal(0,18,clean.shape),0,255).astype(np.uint8)
    rows = []
    for k in [3,5,9,15]:
        result = cv2.GaussianBlur(noisy, (k,k), 0)
        save(f'exercise_kernel_{k}.png', result)
        rows.append({'kernel': k, 'mse': mse(clean,result)})
    report(rows, 'exercise_metrics.json')

if __name__ == '__main__':
    main()
