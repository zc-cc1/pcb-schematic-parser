# 赛题六：PCB电路原理图解析 - Sample Submission

本目录提供了四种提交方式的完整示例，参赛者可以根据自己的技术方案选择合适的方式。

---

## 四种提交方式

### 方式1：纯CV算法（不使用大模型API）

**目录**: `纯CV/`

**适用场景**：
- 使用OpenCV、YOLO、深度学习等传统计算机视觉方法
- 不依赖大模型API
- Token使用量为0

**文件结构**：
```
submission.zip
├── Dockerfile              # Docker构建文件
├── agent.py                # 主程序（当前返回空结果，需实现算法）
└── run.sh                  # 入口脚本（必需，有执行权限）
```

**提交步骤**：
```
按文件结构打包成submission.zip
上传 submission.zip 到评测平台
```

---

### 方式2：LLM方案 - 代码中嵌入API密钥

**目录**: `agent-api/`

**适用场景**：
- 使用阿里云百炼等大模型API
- API密钥直接写在代码中作为fallback
- 适合打榜测试阶段

**特点**：
- API密钥在 `agent.py` 第144行：`os.environ.get('DASHSCOPE_API_KEY', 'sk-...')`
- 即使环境变量未设置，也会使用代码中的密钥

**文件结构**：
```
submission.zip
├── Dockerfile              # Docker构建文件（不含ENV）
├── agent.py                # 主程序（包含API密钥fallback）
├── utils.py                # 工具函数（可选）
└── run.sh                  # 入口脚本（必需，有执行权限）
```

**提交步骤**：
```bash
cd agent-api/
# 修改 agent.py 第144行，替换为你的API密钥
# 按文件结构打包成 submission.zip
# 上传 submission.zip 到评测平台
```

---

### 方式3：LLM方案 - Dockerfile中配置ENV（推荐）

**目录**: `Dockerfile-env-api/`

**适用场景**：
- 使用阿里云百炼等大模型API
- API密钥通过Dockerfile的ENV设置
- 更规范，便于最终评测时切换

**特点**：
- API密钥在 `Dockerfile` 第9行：`ENV DASHSCOPE_API_KEY='sk-...'`
- `agent.py` 从环境变量读取，没有fallback

**文件结构**：
```
submission.zip
├── Dockerfile              # Docker构建文件（包含ENV设置）
├── agent.py                # 主程序（从环境变量读取API密钥）
├── utils.py                # 工具函数（可选）
└── run.sh                  # 入口脚本（必需，有执行权限）
```

**提交步骤**：
```bash
cd Dockerfile-env-api/
# 修改 Dockerfile 第9行，替换为你的API密钥
# 按文件结构打包成 submission.zip
# 上传 submission.zip 到评测平台
```

---

### 方式4：预构建Docker镜像（.tar文件）

**适用场景**：
- 需要自定义基础镜像或复杂依赖
- 包含大型模型文件（如YOLO权重）
- 需要完全控制Docker环境

**文件结构**：
```
submission.tar 
# 这是一个完整的Docker镜像文件
# 必须包含 /submission/run.sh 入口点
# ENTRYPOINT必须设置为 ["/submission/run.sh"]
```

**提交步骤**：

#### 步骤1：构建镜像
```bash
# 从上述任一方式开始构建
cd agent-api/  # 或其他目录
docker build -t my-pcb-parser:v1 .
```

#### 步骤2：导出镜像
```bash
docker save my-pcb-parser:v1 -o submission.tar
```

#### 步骤3：验证并压缩
```bash
# 验证tar文件
docker load -i submission.tar


#### 步骤4：上传
- 将`submission.tar`压缩成`submission.zip`
- 上传 submission.zip到评测平台

**注意事项**：
- 镜像文件可能很大（1-5GB），上传需要时间
- 必须包含 `/submission/run.sh` 入口点
- ENTRYPOINT必须设置为 `["/submission/run.sh"]`

---

## 自定义文件结构

### utils.py是什么？

示例中的 `utils.py` 只是一些**辅助函数示例**（图像预处理、坐标转换等），**不是必需的**。学生可以选择使用或删除。

### 可以有多个文件吗？

**完全可以！** 学生可以按自己的需求组织代码，只需满足：

1. **在Dockerfile中COPY所有文件**
2. **agent.py能正确导入其他模块**
3. **保持入口点为 `/submission/run.sh`**

### 示例：学生的多文件项目

**文件结构**：
```
submission.zip
├── Dockerfile
├── run.sh                  # 入口脚本（必需）
├── agent.py                # 主入口
├── detector.py             # 自定义检测模块
├── parser.py               # 自定义解析模块
├── post_process.py         # 后处理模块
├── config.yaml             # 配置文件
└── models/                 # 模型文件夹
    ├── yolo.pt
    └── weights.pth
```

**对应的Dockerfile**：
```dockerfile
FROM edarace-agent-base:ubuntu24
WORKDIR /submission

# 安装额外依赖
RUN pip3 install --break-system-packages torch ultralytics pyyaml

# 复制所有代码和资源
COPY *.py /submission/
COPY *.yaml /submission/
COPY models/ /submission/models/
COPY run.sh /submission/

RUN chmod +x /submission/run.sh
ENTRYPOINT ["/submission/run.sh"]
```

**agent.py中导入**：
```python
from detector import YOLODetector
from parser import ComponentParser
from post_process import filter_results
```

### 核心要点

- 文件结构完全自由
- 只要agent.py能成功运行即可
- 记得COPY所有需要的文件到/submission/

---

## 输出格式要求

所有提交方式必须生成以下文件：

### 1. 预测结果文件
**路径**: `/output/final_results/{case_id}_Pred.json`

**格式**：
```json
{
  "components": {
    "R1": {
      "bbox": [x1, y1, x2, y2],
      "type": "Resistor",
      "center": [cx, cy]
    }
  },
  "pins": {
    "R1_1": {
      "component": "R1",
      "position": [x, y],
      "pin_number": "1"
    }
  },
  "nets": {
    "net_001": {
      "pins": ["R1_1", "R1_2"],
      "wire_segments": [
        {
          "points": [[x1,y1], [x2,y2]],
          "type": "polyline"
        }
      ]
    }
  }
}
```

### 2. Token使用记录
**路径**: `/output/final_results/decision_log.json`

**格式**：
```json
{
  "method": "qwen-vl-plus",
  "timestamp": "2026-09-28T12:00:00Z",
  "processed_cases": 10,
  "token_usage": {
    "total_tokens": 36882,
    "prompt_tokens": 16882,
    "completion_tokens": 20000,
    "api_calls": 10
  }
}
```

**说明**：
- 如果使用纯CV算法（不使用API），所有token相关字段设为0
- `method` 字段说明使用的方法（如 "opencv", "yolo", "qwen-vl-plus" 等）

---

## 本地测试

### 1. 准备测试数据
```bash
mkdir -p test-input test-output
cp ../最新/10gt/1/1.png test-input/
```

### 2. 构建镜像
```bash
cd agent-api/  # 或其他目录
docker build -t test-pcb:v1 .
```

### 3. 运行测试
```bash
docker run --rm \
  -v $(pwd)/test-input:/input:ro \
  -v $(pwd)/test-output:/output:rw \
  test-pcb:v1 \
  -input /input -output /output
```

### 4. 检查输出
```bash
ls test-output/final_results/
cat test-output/final_results/1_Pred.json
cat test-output/final_results/decision_log.json
```

---

## GPU配置说明

### 评测系统的GPU分配——A800

- **自动分配**：评测系统会自动为每个提交分配GPU
- **日志含义**：`[Ingestion] GPU: 2` 表示使用**2号GPU卡**（不是数量）
- **独占资源**：每个提交独占1张GPU卡
- **无需配置**：学生在Dockerfile中**不需要**设置GPU相关参数

### GPU支持说明

- 所有提交方式都支持GPU加速
- 如果使用纯CV算法不需要GPU，也完全支持（CPU运行）
- GPU ID范围：0-7（8卡服务器）
- 由平台调度器自动管理

### 在代码中使用GPU

如果你的算法需要使用GPU（如PyTorch、TensorFlow），直接使用即可：

```python
import torch

# PyTorch会自动检测可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

**不需要**手动设置 `CUDA_VISIBLE_DEVICES` 或其他GPU环境变量，系统已经配置好了。

---


