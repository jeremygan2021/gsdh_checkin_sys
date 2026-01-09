#!/bin/bash

# =============================================================================
# Docker 镜像构建和部署自动化脚本
# chmod u+x docker_deply.sh
# 用法:
#   ./docker_deply.sh          # 完整构建和部署流程 (默认AMD64架构)
#   ./docker_deply.sh -amd     # 构建和部署AMD64架构
#   ./docker_deply.sh -arm     # 构建和部署ARM64架构
#   ./docker_deply.sh -upload  # 仅上传已存在的tar文件并部署
#   ./docker_deply.sh -upload -amd  # 仅上传已存在的AMD64架构tar文件并部署
#   ./docker_deply.sh -upload -arm  # 仅上传已存在的ARM64架构tar文件并部署
# =============================================================================

# 配置非局域网变量 - 公网上传方法
SERVER_HOST="121.43.104.161"           # 服务器IP地址
SERVER_USER="ubuntu"                     # 服务器用户名
SERVER_PASSWORD="qweasdzxc1"        # 服务器密码
SERVER_PORT="6222"         

# 配置局域网变量 - 公司局域网上传方法
# SERVER_HOST="6.6.6.86"           # 服务器IP地址
# SERVER_USER="ubuntu"                     # 服务器用户名
# SERVER_PASSWORD="qweasdzxc1"        # 服务器密码
# SERVER_PORT="22"                       # SSH端口，默认22


IMAGE_NAME="checkin_sys"              # Docker镜像名称
IMAGE_TAG="latest"                     # Docker镜像标签
CONTAINER_NAME="checkin_sys-container"       # 容器名称
LOCAL_PORT="8800"                      # 本地端口
CONTAINER_PORT="8800"                  # 容器端口
TAR_FILE="${IMAGE_NAME}-${IMAGE_TAG}.tar"  # 压缩包文件名

# 架构相关变量
PLATFORM="linux/amd64"                # 默认架构
ARCH_SUFFIX=""                         # 架构后缀，用于区分不同架构的tar文件
# 默认使用华为云源 (AMD64速度快)
# BASE_IMAGE="swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.9-slim"
# if [ "$PLATFORM" = "linux/arm64" ]; then
#     BASE_IMAGE="docker.m.daocloud.io/python:3.9-slim"
# fi
# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查依赖..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    if ! command -v sshpass &> /dev/null; then
        log_error "sshpass 未安装，请先安装 sshpass"
        log_info "macOS: brew install sshpass"
        log_info "Ubuntu: sudo apt-get install sshpass"
        exit 1
    fi
    
    log_success "依赖检查完成"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -amd)
                PLATFORM="linux/amd64"
                ARCH_SUFFIX="-amd64"
                log_info "设置目标架构为 AMD64"
                shift
                ;;
            -arm)
                PLATFORM="linux/arm64"
                ARCH_SUFFIX="-arm64"
                log_info "设置目标架构为 ARM64"
                shift
                ;;
            -upload)
                UPLOAD_ONLY=true
                log_info "设置为仅上传模式"
                shift
                ;;
            *)
                log_error "未知参数: $1"
                log_info "支持的参数: -amd, -arm, -upload"
                exit 1
                ;;
        esac
    done
    
    # 更新TAR_FILE名，包含架构后缀
    TAR_FILE="${IMAGE_NAME}-${IMAGE_TAG}${ARCH_SUFFIX}.tar"
    log_info "镜像文件名: ${TAR_FILE}"
}

# 构建Docker镜像
build_image() {
    log_info "开始构建 Docker 镜像..."
    
    # 检查是否存在旧的tar文件
    if [ -f "$TAR_FILE" ]; then
        log_warning "发现旧的tar文件，正在删除..."
        rm -f "$TAR_FILE"
    fi
    
    # 构建镜像并导出为tar文件
    docker buildx build --platform $PLATFORM -t "${IMAGE_NAME}:${IMAGE_TAG}" --output type=docker,dest="./${TAR_FILE}" .
    
    if [ $? -eq 0 ]; then
        log_success "Docker 镜像构建完成: ${TAR_FILE}"
    else
        log_error "Docker 镜像构建失败"
        exit 1
    fi
}

# 上传文件到服务器
upload_to_server() {
    log_info "上传文件到服务器..."
    
    # 上传镜像tar文件
    sshpass -p "$SERVER_PASSWORD" scp -P "$SERVER_PORT" -o StrictHostKeyChecking=no "$TAR_FILE" "${SERVER_USER}@${SERVER_HOST}:/tmp/"
    
    # 上传.env文件
    sshpass -p "$SERVER_PASSWORD" scp -P "$SERVER_PORT" -o StrictHostKeyChecking=no ".env" "${SERVER_USER}@${SERVER_HOST}:/tmp/"
    
    if [ $? -eq 0 ]; then
        log_success "文件上传成功"
    else
        log_error "文件上传失败"
        exit 1
    fi
}

# 在服务器上部署
deploy_on_server() {
    log_info "在服务器上部署..."
    
    sshpass -p "$SERVER_PASSWORD" ssh -p "$SERVER_PORT" -o StrictHostKeyChecking=no "${SERVER_USER}@${SERVER_HOST}" << EOF
        set -e
        
        echo "[INFO] 开始服务器端部署..."
        
        # 检查并停止现有容器
        if sudo docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
            echo "[INFO] 发现现有容器 ${CONTAINER_NAME}，正在停止并删除..."
            sudo docker stop ${CONTAINER_NAME} || true
            sudo docker rm ${CONTAINER_NAME} || true
        fi
        
        # 检查并删除现有镜像
        if sudo docker images --format 'table {{.Repository}}:{{.Tag}}' | grep -q "^${IMAGE_NAME}:${IMAGE_TAG}$"; then
            echo "[INFO] 发现现有镜像 ${IMAGE_NAME}:${IMAGE_TAG}，正在删除..."
            sudo docker rmi ${IMAGE_NAME}:${IMAGE_TAG} || true
        fi
        
        # 加载新镜像
        echo "[INFO] 加载新镜像..."
        sudo docker load -i /tmp/${TAR_FILE}
        
        # 验证镜像是否加载成功
        if sudo docker images | grep -q "${IMAGE_NAME}"; then
            echo "[SUCCESS] 镜像加载成功"
        else
            echo "[ERROR] 镜像加载失败"
            exit 1
        fi
        
        # 创建并使用.env文件中的变量
        echo "[INFO] 启动新容器..."
        # 使用 host 网络模式，让容器直接使用宿主机网络栈，从而可以通过 localhost:5432 访问宿主机数据库
        # 同时覆盖环境变量，强制使用本地数据库配置
        sudo docker run -d --network host --name ${CONTAINER_NAME} \
            --env-file /tmp/.env \
            -e DB_HOST=localhost \
            -e DB_PORT=5432 \
            ${IMAGE_NAME}:${IMAGE_TAG}
        
        # 验证容器是否启动成功
        if sudo docker ps | grep -q "${CONTAINER_NAME}"; then
            echo "[SUCCESS] 容器启动成功"
            echo "[INFO] 容器状态:"
            sudo docker ps | grep "${CONTAINER_NAME}"
        else
            echo "[ERROR] 容器启动失败"
            echo "[INFO] 查看容器日志:"
            sudo docker logs ${CONTAINER_NAME}
            exit 1
        fi
        
        # 清理临时文件
        echo "[INFO] 清理临时文件..."
        rm -f /tmp/${TAR_FILE}
        rm -f /tmp/.env
        
        echo "[SUCCESS] 部署完成！"
        echo "[INFO] 应用访问地址: http://${SERVER_HOST}:${LOCAL_PORT}"
EOF

    if [ $? -eq 0 ]; then
        log_success "服务器部署完成"
    else
        log_error "服务器部署失败"
        exit 1
    fi
}

# 清理本地文件
cleanup_local() {
    log_info "清理本地临时文件..."
    
    if [ -f "$TAR_FILE" ]; then
        rm -f "$TAR_FILE"
        log_success "本地临时文件已清理"
    fi
}

# 显示部署信息
show_deployment_info() {
    log_success "部署完成！"
    echo ""
    echo "=========================================="
    echo "部署信息:"
    echo "=========================================="
    echo "服务器地址: ${SERVER_HOST}"
    echo "应用端口: ${LOCAL_PORT}"
    echo "访问地址: http://${SERVER_HOST}:${LOCAL_PORT}"
    echo "容器名称: ${CONTAINER_NAME}"
    echo "镜像名称: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "目标架构: ${PLATFORM}"
    echo "镜像文件: ${TAR_FILE}"
    echo "=========================================="
    echo ""
    log_info "如需查看容器日志，请在服务器上运行: sudo docker logs ${CONTAINER_NAME}"
    log_info "如需停止容器，请在服务器上运行: sudo docker stop ${CONTAINER_NAME}"
}

# 主函数
main() {
    echo "=========================================="
    echo "Docker 镜像构建和部署自动化脚本"
    echo "=========================================="
    echo ""
    
    # 解析命令行参数
    UPLOAD_ONLY=false
    parse_arguments "$@"
    
    # 检查配置
    if [ "$SERVER_HOST" = "your-server-ip" ] || [ "$SERVER_PASSWORD" = "your-password" ]; then
        log_error "请先配置脚本顶部的服务器信息"
        log_info "需要修改的变量:"
        log_info "  - SERVER_HOST: 服务器IP地址"
        log_info "  - SERVER_USER: 服务器用户名"
        log_info "  - SERVER_PASSWORD: 服务器密码"
        exit 1
    fi
    
    # 检查是否是上传模式
    if [ "$UPLOAD_ONLY" = true ]; then
        log_info "检测到 -upload 参数，跳过构建步骤"
        
        # 检查tar文件是否存在
        if [ ! -f "$TAR_FILE" ]; then
            log_error "未找到tar文件: $TAR_FILE"
            log_info "请先运行脚本构建镜像，或确保tar文件存在"
            exit 1
        fi
        
        log_success "找到tar文件: $TAR_FILE"
        
        # 执行上传和部署流程
        upload_to_server
        deploy_on_server
        cleanup_local
        show_deployment_info
    else
        # 执行完整的部署流程
        check_dependencies
        build_image
        upload_to_server
        deploy_on_server
        cleanup_local
        show_deployment_info
    fi
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi