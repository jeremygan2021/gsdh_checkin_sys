# 使用 DaoCloud 国内镜像代理加速下载 (支持多架构)
# 接收构建参数 BASE_IMAGE，由 docker_deply.sh 传入
ARG BASE_IMAGE=python:3.9-slim
FROM ${BASE_IMAGE}


# 设置工作目录
WORKDIR /app

# 设置 pip 国内镜像源（清华源）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置环境变量
# 防止 Python 生成 .pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1
# 确保 Python 输出不被缓冲
ENV PYTHONUNBUFFERED=1

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8800

# 启动命令
CMD ["python", "main.py", "--port", "8800"]
