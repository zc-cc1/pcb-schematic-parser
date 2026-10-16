#include "image_utils.hpp"
#include <iostream>

int main() {
    try {
        cv::Mat board(160, 240, CV_8UC3, cv::Scalar(0, 0, 0));
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 6; ++col) {
                if ((row + col) % 2 == 0) {
                    board(cv::Rect(col * 40, row * 40, 40, 40)).setTo(cv::Scalar(255, 255, 255));
                }
            }
        }
        int white_pixels = 0;
        for (int y = 0; y < board.rows; ++y) {
            for (int x = 0; x < board.cols; ++x) {
                const auto p = board.at<cv::Vec3b>(y, x);
                if (p[0] == 255 && p[1] == 255 && p[2] == 255) ++white_pixels;
            }
        }
        const cv::Mat darker = lesson::brighten(board, -50);
        lesson::require(white_pixels == 19200, "Expected 19200 white pixels");
        lesson::require(darker.at<cv::Vec3b>(0, 0)[0] == 205, "White should become 205");
        lesson::require(darker.at<cv::Vec3b>(0, 40)[0] == 0, "Black should remain zero");
        lesson::save("exercise_checkerboard.png", board);
        lesson::save("exercise_darker.png", darker);
        lesson::write_text("exercise_metrics.json", "{\n  \"white_pixel_count\": 19200,\n  \"checks_passed\": true\n}\n");
        std::cout << "White pixels=" << white_pixels << "; all checks passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
