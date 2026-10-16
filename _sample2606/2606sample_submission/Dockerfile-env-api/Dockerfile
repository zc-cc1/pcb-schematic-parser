FROM edarace-agent-base:ubuntu24

WORKDIR /submission

# 安装OpenAI库（用于调用阿里云百炼API）
RUN pip3 install --break-system-packages openai

# 设置API密钥（打榜阶段使用，最终评测前删除此行）
ENV DASHSCOPE_API_KEY='sk-your-key'

# 复制代码文件
COPY agent.py /submission/
COPY utils.py /submission/
COPY run.sh /submission/

# 设置执行权限
RUN chmod +x /submission/run.sh

# 设置入口点
ENTRYPOINT ["/submission/run.sh"]
