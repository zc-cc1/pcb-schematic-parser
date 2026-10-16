from demo import np, save, report, brighten

def main():
    board = np.zeros((160, 240, 3), np.uint8)
    for row in range(4):
        for col in range(6):
            if (row + col) % 2 == 0:
                board[row*40:(row+1)*40, col*40:(col+1)*40] = 255
    save('exercise_checkerboard.png', board)
    save('exercise_darker.png', brighten(board, -50))
    report({'white_pixel_count': int(np.count_nonzero(board[:, :, 0] == 255)),
            'expected_white_pixel_count': 19200}, 'exercise_metrics.json')

if __name__ == '__main__':
    main()
