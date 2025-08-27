# Docker 部署指南

本文档介绍如何使用 Docker 和 Docker Compose 部署 Z2API 服务。

## 快速开始

### 1. 环境准备

确保已安装：

- Docker (>= 20.10)
- Docker Compose (>= 2.0)

### 2. 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置必要的环境变量：

```bash
# 必须设置的变量
API_KEY=your-api-key-here
Z_AI_COOKIES=your_jwt_token_here

# 可选配置
HOST=0.0.0.0
PORT=3007
LOG_LEVEL=INFO
```

### 3. 构建和运行

#### 方式一：使用 Docker Compose（推荐）

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

#### 方式二：使用 Docker 直接运行

```bash
# 构建镜像
docker build -t z2api .

# 运行容器
docker run -d \
  --name z2api \
  -p 3007:3007 \
  --env-file .env \
  z2api
```

### 4. 验证部署

访问健康检查端点：

```bash
curl http://localhost:3007/health
```

## 高级配置

### 使用 Nginx 反向代理

启用 Nginx 反向代理（可选）：

```bash
docker-compose --profile with-nginx up -d
```

这将启动：

- Z2API 服务（内部端口 3007）
- Nginx 反向代理（端口 80/443）

### 自定义配置

#### 修改 Nginx 配置

编辑 `nginx.conf` 文件来自定义 Nginx 设置。

#### 添加 SSL 证书

将 SSL 证书放在 `ssl/` 目录下：

```
ssl/
├── cert.pem
└── key.pem
```

然后取消注释 `nginx.conf` 中的 HTTPS 配置。

### 生产环境部署

#### 1. 使用预构建镜像

```bash
# 拉取最新镜像
docker pull ghcr.io/your-username/z2api:latest

# 使用预构建镜像运行
docker run -d \
  --name z2api \
  -p 3007:3007 \
  --env-file .env \
  --restart unless-stopped \
  ghcr.io/your-username/z2api:latest
```

#### 2. 资源限制

在生产环境中，建议设置资源限制：

```yaml
# docker-compose.override.yml
version: "3.8"
services:
  z2api:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
        reservations:
          cpus: "0.5"
          memory: 256M
```

#### 3. 日志管理

配置日志轮转：

```yaml
services:
  z2api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## GitHub Actions CI/CD

项目包含自动化的 CI/CD 流程：

### 自动构建

- 推送到 `main`、`master` 或 `develop` 分支时自动构建
- 创建标签时构建带版本号的镜像
- 支持多架构构建（amd64, arm64）

### 安全扫描

- 使用 Trivy 进行容器安全扫描
- 扫描结果上传到 GitHub Security tab

### 使用方法

1. 确保 GitHub 仓库启用了 GitHub Packages
2. 推送代码到仓库
3. 查看 Actions 页面的构建状态
4. 构建成功后，镜像将推送到 `ghcr.io/your-username/repo-name`

## 故障排除

### 常见问题

#### 1. 容器启动失败

```bash
# 查看容器日志
docker-compose logs z2api

# 检查容器状态
docker-compose ps
```

#### 2. 健康检查失败

```bash
# 进入容器检查
docker-compose exec z2api bash

# 手动测试健康检查
curl localhost:3007/health
```

#### 3. 环境变量问题

```bash
# 检查环境变量是否正确加载
docker-compose exec z2api env | grep Z_AI
```

### 性能优化

#### 1. 启用多进程

修改 Dockerfile 中的启动命令：

```dockerfile
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3007", "--workers", "4"]
```

#### 2. 使用更小的基础镜像

```dockerfile
FROM python:3.11-alpine
```

## 监控和日志

### 日志收集

日志默认输出到容器标准输出，可以通过以下方式查看：

```bash
# 实时查看日志
docker-compose logs -f z2api

# 查看最近的日志
docker-compose logs --tail=100 z2api
```

### 健康监控

服务提供健康检查端点 `/health`，可以集成到监控系统中。

## 安全建议

1. **不要在镜像中包含敏感信息**
2. **使用非 root 用户运行容器**（已在 Dockerfile 中配置）
3. **定期更新基础镜像**
4. **使用安全扫描工具**（已集成 Trivy）
5. **限制容器权限**
6. **使用 HTTPS**（生产环境）

## 支持

如有问题，请查看：

1. 项目 README.md
2. GitHub Issues
3. 容器日志输出
