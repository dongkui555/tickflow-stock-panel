FROM python:3.11-slim

ARG USE_CN_MIRROR=1
ARG PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    TDX_API_HOST=0.0.0.0 \
    TDX_API_PORT=8080

WORKDIR /tdx-api

COPY requirements.txt .
RUN if [ "$USE_CN_MIRROR" = "1" ]; then \
      pip install --no-cache-dir -i "$PYPI_INDEX" -r requirements.txt; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi

COPY app.py .

EXPOSE 8080
CMD ["python", "app.py"]
