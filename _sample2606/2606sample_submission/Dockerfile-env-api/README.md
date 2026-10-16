# LLM方案 - Dockerfile中配置ENV（推荐）

使用大模型API，API密钥通过Dockerfile的ENV设置。**这是推荐的方式**。

## 快速开始

### 1. 修改API密钥
编辑 `Dockerfile` 第9行：
```dockerfile
ENV DASHSCOPE_API_KEY='sk-你的API密钥'
```

### 2. 打包提交
```bash
zip submission.zip Dockerfile agent.py utils.py run.sh
```

### 3. 上传到评测平台

## 文件说明

- `Dockerfile` - Docker构建文件，**第9行包含ENV设置**
- `agent.py` - 主程序，从环境变量读取密钥（无fallback）
- `utils.py` - 工具函数
- `run.sh` - 入口脚本

## 本地测试

```bash
docker build -t test:v1 .
docker run --rm \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output:rw \
  test:v1 -input /input -output /output
```


## 与agent-api的区别

| 特性 | Dockerfile-env-api（本方式） | agent-api |
|------|--------------------------|-----------|
| 密钥位置 | Dockerfile | 代码中 |
| 有fallback | 否（更严格） | 是 |
| 最终评测 | 只改Dockerfile | 需改代码 |

## 优化建议

- **提示词**：修改 `agent.py` 第28-60行
- **温度参数**：第90行已设为0（确定性输出）


---

更多信息见上级目录 [README.md](../README.md)
