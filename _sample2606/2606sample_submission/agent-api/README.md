# LLM方案 - 代码中嵌入API密钥

使用大模型API，API密钥直接写在代码中作为fallback。适合快速测试。

## 快速开始

### 1. 修改API密钥
编辑 `agent.py` 第144行：
```python
api_key = os.environ.get('DASHSCOPE_API_KEY', 'sk-你的API密钥')
```

### 2. 打包提交
```bash
zip submission.zip Dockerfile agent.py utils.py run.sh
```

### 3. 上传到评测平台

## 文件说明

- `agent.py` - 主程序，**第144行包含API密钥fallback**
- `Dockerfile` - Docker构建文件（不含ENV）
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

## 优化建议

- **提示词**：修改 `agent.py` 第28-60行
- **温度参数**：第90行已设为0（确定性输出）

## 注意事项

 **最终评测前需删除代码中的密钥**（等组委会通知）

---

更多信息见上级目录 [README.md](../README.md)
