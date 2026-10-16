// 纯C++演示：不依赖OpenCV，先练习变量、分支、循环、函数。
#include <cstdint>
#include <iostream>

int add(int a, int b) {
    return a + b;
}

int main() {
    const int width = 300;  // const表示之后不能重新赋值
    const int height = 180;
    int values[] = {10, 20, 30};
    for (int i = 0; i < 3; ++i) {
        if (values[i] > 15) {
            std::cout << "value=" << values[i] << '\n';
        }
    }
    const std::uint8_t a = 250;
    const std::uint8_t b = 10;
    const int sum = a + b;  // 先提升为int，所以这里是260，不是4！
    const auto narrowed = static_cast<std::uint8_t>(sum); // 转回8位无符号才成为4
    std::cout << "width=" << width << ", height=" << height << '\n';
    std::cout << "add(3,5)=" << add(3, 5) << '\n';
    std::cout << "int sum=" << sum << ", narrowed=" << static_cast<int>(narrowed) << '\n';
    // 返回非零值让CTest知道检查失败；不用assert，避免Release构建关闭检查。
    return (add(3, 5) == 8 && sum == 260 && narrowed == 4) ? 0 : 1;
}
