# ===== 安全培训智能 Agent CPU 镜像 =====
FROM python:3.12-slim

# 系统依赖（国内 apt 源加速）：
# - LibreOffice（writer/impress）：Linux 下把 .doc/.ppt 转换为 .docx/.pptx
# - 中文字体、OpenCV/ONNX 运行库（rapidocr）、poppler（PDF）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' \
        /etc/apt/sources.list.d/debian.sources || true \
    && apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer \
        libreoffice-impress \
        fonts-wqy-zenhei \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# 先单独安装 CPU 版 torch：PyPI 默认 Linux wheel 会把数 GB 的 nvidia-* CUDA 依赖一并拉入
# BuildKit 缓存挂载：失败重试时复用已下载的 wheel，避免重复下载
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.0
# 其余依赖走阿里云 PyPI 源；requirements 中 torch 带 Windows 平台标记，Linux 下自动跳过
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        -r requirements.txt

COPY . .

EXPOSE 8000

# 容器内默认服务；数据目录、模型缓存、向量库通过 docker-compose 挂载
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
