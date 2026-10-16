# 纯CV算法示例

这是一个**不使用大模型API的纯CV算法**提交示例。

## 特点

- 不调用任何API（token使用量为0）
- 不需要API密钥
- 适合使用OpenCV、YOLO、深度学习等CV方法

## 文件说明

- `Dockerfile` - Docker构建文件
- `agent.py` - 主程序（当前返回空结果，需要实现算法）
- `run.sh` - 入口脚本

## 快速开始

### 1. 实现你的算法

编辑 `agent.py`，在 `main()` 函数中实现你的CV算法：

```python
# 示例：使用OpenCV处理
import cv2
import numpy as np

for img in images:
    # 读取图像
    image = cv2.imread(str(img))
    
    # 你的CV算法
    # 1. 预处理
    # 2. 检测元件
    # 3. 识别引脚
    # 4. 追踪连线
    
    result = {
        "components": {...},
        "pins": {...},
        "nets": {...}
    }
```

### 2. 本地测试

```bash
docker build -t cv-test:v1 .

docker run --rm \
  -v /path/to/test-input:/input:ro \
  -v /path/to/test-output:/output:rw \
  cv-test:v1 \
  -input /input -output /output
```

### 3. 打包提交

```bash
zip submission.zip Dockerfile agent.py run.sh
# 上传 submission.zip 到评测平台
```

## 注意事项

- 输出必须符合JSON格式规范（见主README）
- 必须生成 `decision_log.json`（token字段设为0）

---

更多信息请参考上级目录的 [README.md](../README.md)
