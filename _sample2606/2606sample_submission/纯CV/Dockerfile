FROM edarace-agent-base:ubuntu24

WORKDIR /submission

# 复制代码
COPY agent.py /submission/
COPY run.sh /submission/

# 设置执行权限
RUN chmod +x /submission/run.sh

# 设置入口点
ENTRYPOINT ["/submission/run.sh"]
