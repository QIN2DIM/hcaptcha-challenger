FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy as builder

WORKDIR /app

# 安装必要的系统依赖包
RUN apt-get update && apt-get install -y \
    xvfb \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

# 安装 tini 作为 init 系统
RUN apt-get update && apt-get install -y tini

# 设置 tini 作为容器入口点
ENTRYPOINT ["/usr/bin/tini", "--"]

# 在容器中启动 headted browser
ENV DISPLAY=:99

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install undetected-playwright-patch

# 设置playwright相关文件的权限
RUN chmod -R 755 /usr/local/lib/python3.10/dist-packages/undetected_playwright/driver/ && \
    chmod +x /usr/local/lib/python3.10/dist-packages/undetected_playwright/driver/playwright.sh && \
    chmod +x /usr/local/lib/python3.10/dist-packages/undetected_playwright/driver/node

# 确保文件所有权正确
RUN chown -R root:root /usr/local/lib/python3.10/dist-packages/undetected_playwright/