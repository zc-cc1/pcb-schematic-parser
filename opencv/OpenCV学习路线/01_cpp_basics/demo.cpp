#include "image_utils.hpp"
#include <cstdint>
#include <iostream>
#include <sstream>

int main() {
    try {
        // Mat参数先行后列：高180、宽300；CV_8UC3为8位无符号3通道。
        cv::Mat image(180, 300, CV_8UC3, cv::Scalar(40, 40, 40));
        const cv::Scalar colors[] = {
            cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)
        }; // BGR：蓝、绿、红
        for (int i = 0; i < 3; ++i) {
            const int left = 20 + i * 90;
            image(cv::Rect(left, 40, 70, 100)).setTo(colors[i]);
        }
        cv::Mat crop = image(cv::Rect(20, 40, 70, 100)).clone();
        cv::Mat shared = image;   // 浅拷贝：仍指向image的像素
        cv::Mat copied = image.clone();  // 深拷贝：独立像素
        cv::Mat view = copied(cv::Rect(0, 0, 20, 20));
        view.setTo(cv::Scalar(255, 255, 255)); // copied改变，image不变

        lesson::save("01_colors.png", image);
        lesson::save("02_crop.png", crop);
        lesson::save("03_bright.png", lesson::brighten(image, 60));
        lesson::save("04_view_changed_parent.png", copied);

        const std::uint8_t a = 250, b = 10;
        const int sum = a + b;  // 整数提升：260
        const int narrowed = static_cast<std::uint8_t>(sum); // 缩窄后4
        const int clipped = cv::saturate_cast<unsigned char>(sum); // 饱和后255
        const cv::Vec3b pixel = image.at<cv::Vec3b>(50, 30); // y=50,x=30
        lesson::require(image.rows == 180 && image.cols == 300 && image.channels() == 3, "Image shape mismatch");
        lesson::require(crop.rows == 100 && crop.cols == 70, "Crop shape mismatch");
        lesson::require(pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0, "Expected blue pixel");
        lesson::require(shared.data == image.data && copied.data != image.data, "Copy semantics mismatch");
        lesson::require(copied.at<cv::Vec3b>(0, 0)[0] == 255 && image.at<cv::Vec3b>(0, 0)[0] == 40, "ROI sharing mismatch");
        lesson::require(sum == 260 && narrowed == 4 && clipped == 255, "Numeric conversion mismatch");
        std::ostringstream metrics;
        metrics << "{\n  \"language\": \"C++\",\n  \"rows\": 180,\n  \"cols\": 300,\n"
                << "  \"channels\": 3,\n  \"int_sum\": " << sum
                << ",\n  \"narrowed_uint8\": " << narrowed
                << ",\n  \"saturated_uint8\": " << clipped << ",\n  \"checks_passed\": true\n}\n";
        lesson::write_text("metrics.json", metrics.str());
        std::cout << metrics.str() << "Saved images in this week's outputs folder.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
