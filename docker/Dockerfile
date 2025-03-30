FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy as builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    xvfb \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y tini

ENTRYPOINT ["/usr/bin/tini", "--"]

ENV DISPLAY=:99

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install undetected-playwright-patch

RUN chmod -R 755 /usr/local/lib/python3.10/dist-packages/undetected_playwright/driver/ && \
    chmod +x /usr/local/lib/python3.10/dist-packages/undetected_playwright/driver/playwright.sh && \
    chmod +x /usr/local/lib/python3.10/dist-packages/undetected_playwright/driver/node

RUN chown -R root:root /usr/local/lib/python3.10/dist-packages/undetected_playwright/