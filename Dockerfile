FROM ubuntu:22.04

# 避免 apt-get 安装时的交互提示
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# 仅安装核心编译所需的依赖，并清理缓存以缩减镜像体积
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 将当前目录下的代码拷贝到容器中
COPY . .

# 赋予压测脚本可执行权限
RUN chmod +x run_cache_benchmark.sh

# 设定入口点为压测脚本
ENTRYPOINT ["./run_cache_benchmark.sh"]

# 默认运行参数 [头文件路径, 操作数, 缓存容量, 线程数, 分片数]
# 当使用 docker-compose 时，这里的 CMD 会被 docker-compose.yml 中的 command 覆盖
CMD [".", "5000000", "16384", "8", "8"]