#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lesson {
// CMake传入UTF-8路径，再交给filesystem处理Windows中文目录。
inline std::filesystem::path output_dir() {
    auto path = std::filesystem::u8path(WEEK01_OUTPUT_DIR);
    std::filesystem::create_directories(path);
    return path;
}

inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

inline void save(const std::string& name, const cv::Mat& image) {
    std::vector<unsigned char> bytes;
    require(cv::imencode(".png", image, bytes), "PNG encoding failed");
    // imencode只处理图像；文件路径由C++17文件流处理，避免窄字符串路径兼容问题。
    std::ofstream out(output_dir() / name, std::ios::binary);
    require(static_cast<bool>(out), "Cannot open output image");
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    out.close();
    require(static_cast<bool>(out), "Cannot finish writing image");
}

inline void write_text(const std::string& name, const std::string& text) {
    std::ofstream out(output_dir() / name);
    require(static_cast<bool>(out), "Cannot open report");
    out << text;
    out.close();
    require(static_cast<bool>(out), "Cannot finish writing report");
}

inline cv::Mat brighten(const cv::Mat& image, int delta) {
    require(image.type() == CV_8UC3, "brighten expects CV_8UC3");
    cv::Mat result = image.clone();
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            cv::Vec3b& target = result.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                // 像素先按int运算，再饱和裁剪到0～255，避免转回8位时回绕。
                target[c] = cv::saturate_cast<unsigned char>(static_cast<int>(pixel[c]) + delta);
            }
        }
    }
    return result;
}
} // namespace lesson
