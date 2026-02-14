#!/bin/bash

# macOS Op-Plugin 一键编译和测试脚本
# 用于在 macOS 上编译和测试 MindSpore op-plugin

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $*"
}

# 获取最新的 MindSpore daily 版本
get_latest_mindspore_daily() {
    local python_version="${1:-39}"  # 默认 Python 3.9
    local target_arch="x86_64"
    
    log_info "查找最新的 MindSpore daily 版本..."
    
    # 获取版本列表页面
    local version_page
    version_page=$(curl -s "${MINDSPORE_REPO_BASE}/" 2>/dev/null || echo "")
    
    if [ -z "$version_page" ]; then
        log_error "无法访问 MindSpore 版本页面"
        return 1
    fi
    
    # 提取最新的日期目录（格式：YYYYMM，例如 202601）
    local latest_date_dir
    latest_date_dir=$(echo "$version_page" | grep -oE '<[0-9]{6}/>' | sed 's/[<>]//g' | sort -r | head -1)
    
    # 如果上面的方法没找到，尝试从链接中提取
    if [ -z "$latest_date_dir" ]; then
        latest_date_dir=$(echo "$version_page" | grep -oE 'href="[0-9]{6}/"' | sed 's/href="//;s/"//;s/\/$//' | sort -r | head -1)
    fi
    
    # 如果还是没找到，尝试更宽松的匹配
    if [ -z "$latest_date_dir" ]; then
        latest_date_dir=$(echo "$version_page" | grep -oE '<[0-9]{6}>' | sed 's/[<>]//g' | sort -r | head -1)
    fi
    
    # 更宽松的匹配，提取所有6位数字
    if [ -z "$latest_date_dir" ]; then
        latest_date_dir=$(echo "$version_page" | grep -oE '[0-9]{6}' | grep -E '^20[0-9]{4}$' | sort -r | head -1)
    fi
    
    if [ -z "$latest_date_dir" ]; then
        log_error "未找到版本目录"
        return 1
    fi
    
    # 获取该目录下的日期子目录（格式：YYYYMMDD）
    local date_page
    date_page=$(curl -s "${MINDSPORE_REPO_BASE}/${latest_date_dir}/" 2>/dev/null || echo "")
    
    if [ -z "$date_page" ]; then
        log_error "无法访问日期目录"
        return 1
    fi
    
    # 提取所有日期子目录（格式：YYYYMMDD）
    local all_dates
    all_dates=$(echo "$date_page" | grep -oE '<[0-9]{8}/>' | sed 's/[<>]//g' | sort -r)
    
    # 如果上面的方法没找到，尝试从链接中提取
    if [ -z "$all_dates" ]; then
        all_dates=$(echo "$date_page" | grep -oE 'href="[0-9]{8}/"' | sed 's/href="//;s/"//;s/\/$//' | sort -r)
    fi
    
    # 如果还是没找到，尝试更宽松的匹配
    if [ -z "$all_dates" ]; then
        all_dates=$(echo "$date_page" | grep -oE '[0-9]{8}' | grep -E '^20[0-9]{6}$' | sort -r)
    fi
    
    if [ -z "$all_dates" ]; then
        log_error "未找到日期子目录"
        return 1
    fi
    
    # 从当前日期开始向前回溯最多10天，查找最新的可用版本
    local target_date
    target_date=$(python3 -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d'))" 2>/dev/null)
    
    if [ -z "$target_date" ]; then
        log_error "无法获取当前日期"
        return 1
    fi
    
    local dates_to_try=()
    
    # 从当前日期开始向前回溯最多10天，收集所有可用的日期
    for i in $(seq 0 9); do
        # 使用 Python 计算日期（跨平台兼容）
        local check_date=$(python3 -c "
from datetime import datetime, timedelta
try:
    target = datetime.strptime('$target_date', '%Y%m%d')
    check = target - timedelta(days=$i)
    print(check.strftime('%Y%m%d'))
except:
    pass
" 2>/dev/null)
        
        # 检查计算出的日期是否在可用日期列表中
        if [ -n "$check_date" ] && echo "$all_dates" | grep -q "^${check_date}$"; then
            local date_exists=false
            for existing_date in "${dates_to_try[@]}"; do
                if [ "$existing_date" = "$check_date" ]; then
                    date_exists=true
                    break
                fi
            done
            if [ "$date_exists" = "false" ]; then
                dates_to_try+=("$check_date")
            fi
        fi
    done
    
    # 如果回溯也没找到，使用最新的日期
    if [ ${#dates_to_try[@]} -eq 0 ]; then
        local fallback_date=$(echo "$all_dates" | head -1)
        dates_to_try+=("$fallback_date")
        log_warn "未找到目标日期附近的有效日期，使用最新日期: $fallback_date"
    fi
    
    # 获取该日期下的 master 目录，如果找不到则回溯查找
    local master_dir=""
    local found_whl=false
    local whl_file=""
    local download_base=""
    local latest_subdir=""
    
    # 尝试每个日期，找到第一个有 master 目录且有对应 whl 文件的
    for try_date in "${dates_to_try[@]}"; do
        local master_page
        master_page=$(curl -s "${MINDSPORE_REPO_BASE}/${latest_date_dir}/${try_date}/" 2>/dev/null || echo "")
        
        if [ -z "$master_page" ]; then
            continue
        fi
        
        # 提取 master 目录名
        # 方法1: 匹配 <master_xxx/> 格式
        master_dir=$(echo "$master_page" | grep -oE '<master_[^/<>]+/>' | sed 's/[<>]//g' | sed 's/\/$//' | head -1)
        
        # 方法2: 从链接中提取 href="master_xxx/"
        if [ -z "$master_dir" ]; then
            master_dir=$(echo "$master_page" | grep -oE 'href="master_[^"]+/"' | sed 's/href="//;s/"//;s/\/$//' | head -1)
        fi
        
        # 方法3: 匹配表格中的链接格式
        if [ -z "$master_dir" ]; then
            master_dir=$(echo "$master_page" | grep -oE 'master_[a-zA-Z0-9_]+' | head -1)
        fi
        
        # 方法4: 更宽松的匹配
        if [ -z "$master_dir" ]; then
            master_dir=$(echo "$master_page" | grep -oE 'master_[^<>\s"]+' | head -1)
        fi
        
        # 方法5: 从完整的链接中提取
        if [ -z "$master_dir" ]; then
            master_dir=$(echo "$master_page" | grep -oE '/master_[^/]+/' | sed 's/\///g' | head -1)
        fi
        
        # 确保移除末尾斜杠
        master_dir="${master_dir%/}"
        
        if [ -z "$master_dir" ]; then
            continue
        fi
        
        # 构建完整的下载路径
        download_base="${MINDSPORE_REPO_BASE}/${latest_date_dir}/${try_date}/${master_dir}/cpu/${target_arch}/"
        
        # 获取文件列表
        local file_page
        file_page=$(curl -s "$download_base" 2>/dev/null || echo "")
        
        if [ -z "$file_page" ]; then
            continue
        fi
        
        # 查找匹配的 whl 文件
        # 注意：HTML 中可能包含 title 属性，需要清理
        whl_file=$(echo "$file_page" | grep -oE "mindspore-[0-9]+\.[0-9]+\.[0-9]+-cp${python_version}-cp${python_version}-macosx[^<\"\s]*x86_64\.whl" | head -1)
        
        # 如果找到，清理 HTML 标签和 URL 编码
        if [ -n "$whl_file" ]; then
            # 移除可能的 HTML 标签和属性（如 title="..."）
            whl_file=$(echo "$whl_file" | sed 's/"[^"]*"//g' | sed 's/>[^<]*//g' | sed 's/<[^>]*>//g')
            # 解码 URL 编码
            whl_file=$(echo "$whl_file" | sed 's/%5F/_/g' | sed 's/%2D/-/g')
            # 移除可能的引号和空格
            whl_file=$(echo "$whl_file" | tr -d '"' | tr -d ' ' | tr -d '\n')
        fi
        
        # 如果还是没找到，尝试更宽松的匹配
        if [ -z "$whl_file" ]; then
            whl_file=$(echo "$file_page" | grep -oE "mindspore-[0-9]+\.[0-9]+\.[0-9]+-cp${python_version}-cp${python_version}-macosx[^<\"\s]*\.whl" | grep "x86_64" | head -1)
            if [ -n "$whl_file" ]; then
                # 清理 HTML 标签
                whl_file=$(echo "$whl_file" | sed 's/"[^"]*"//g' | sed 's/>[^<]*//g' | sed 's/<[^>]*>//g')
                # 解码 URL 编码
                whl_file=$(echo "$whl_file" | sed 's/%5F/_/g' | sed 's/%2D/-/g')
                # 移除可能的引号和空格
                whl_file=$(echo "$whl_file" | tr -d '"' | tr -d ' ' | tr -d '\n')
            fi
        fi
        
        if [ -n "$whl_file" ]; then
            latest_subdir="$try_date"  # 更新使用的日期
            found_whl=true
            break
        fi
    done
    
    if [ "$found_whl" != "true" ] || [ -z "$master_dir" ] || [ -z "$whl_file" ]; then
        log_error "未找到有效的 daily 版本（已尝试 ${#dates_to_try[@]} 个日期）"
        return 1
    fi
    
    # 构建完整 URL
    local whl_url="${download_base}${whl_file}"
    
    # 提取版本号（从文件名中）
    local version
    version=$(echo "$whl_file" | grep -oE 'mindspore-[0-9]+\.[0-9]+\.[0-9]+' | cut -d'-' -f2)
    
    log_info "✅ 找到 daily 版本: $version (日期: $latest_subdir)"
    
    # 返回结果（通过全局变量）
    MINDSPORE_DAILY_URL="$whl_url"
    MINDSPORE_DAILY_VERSION="$version"
    MINDSPORE_DAILY_FILE="$whl_file"
    
    return 0
}

# 配置
ENV_NAME="${1:-py39_x86_ms_op_plugin}"
MINDSPORE_VERSION="${2:-daily}"  # 默认使用 daily 版本
USE_DAILY="${USE_DAILY:-true}"   # 是否使用 daily 版本
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_ENV_CREATE="${SKIP_ENV_CREATE:-false}"
SKIP_BUILD="${SKIP_BUILD:-false}"
SKIP_TEST="${SKIP_TEST:-false}"

# MindSpore 仓库配置
MINDSPORE_REPO_BASE="https://repo.mindspore.cn/mindspore/mindspore/version"

echo "=========================================="
echo "  macOS Op-Plugin 一键编译和测试脚本"
echo "=========================================="
echo ""
log_info "项目目录: $PROJECT_DIR"
log_info "环境名称: $ENV_NAME"
if [ "$MINDSPORE_VERSION" = "daily" ] || [ "$USE_DAILY" = "true" ]; then
    log_info "MindSpore 版本: daily (将自动获取最新版本)"
else
    log_info "MindSpore 版本: $MINDSPORE_VERSION"
fi
echo ""

# 步骤 1: 创建 x86_64 conda 环境
ENV_EXISTS=false
if conda env list | grep -q "^${ENV_NAME} "; then
    ENV_EXISTS=true
fi

if [ "$SKIP_ENV_CREATE" != "true" ]; then
    log_step "步骤 1: 创建 x86_64 conda 环境"
    
    # 检查环境是否已存在
    if [ "$ENV_EXISTS" = "true" ]; then
        log_warn "环境 $ENV_NAME 已存在"
        log_info "将使用现有环境，进行更新"
    else
        log_info "创建 x86_64 conda 环境（使用默认源）..."
        CONDA_SUBDIR=osx-64 conda create -n "$ENV_NAME" python=3.9 -y \
            --override-channels -c defaults -c conda-forge || {
            log_warn "使用默认源创建失败，尝试仅使用 conda-forge..."
            CONDA_SUBDIR=osx-64 conda create -n "$ENV_NAME" python=3.9 -y -c conda-forge || {
                log_error "创建环境失败"
                log_info "请手动执行:"
                log_info "  CONDA_SUBDIR=osx-64 conda create -n $ENV_NAME python=3.9 -y --override-channels -c defaults"
                exit 1
            }
        }
        log_info "✅ 环境创建成功"
    fi
else
    log_info "跳过环境创建（SKIP_ENV_CREATE=true）"
fi

# 激活环境
log_step "步骤 2: 激活并配置环境"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || {
    log_error "激活环境失败"
    exit 1
}

# 确保使用 x86_64
conda config --env --set subdir osx-64 -q

# 验证架构
PYTHON_ARCH=$(python -c "import platform; print(platform.machine())" 2>/dev/null)
if [ "$PYTHON_ARCH" != "x86_64" ]; then
    log_warn "Python 架构不是 x86_64，尝试重新安装..."
    conda install python=3.9 -y -q --force-reinstall
    PYTHON_ARCH=$(python -c "import platform; print(platform.machine())" 2>/dev/null)
    if [ "$PYTHON_ARCH" != "x86_64" ]; then
        log_error "❌ Python 架构不匹配，无法继续"
        exit 1
    fi
fi

# 步骤 3: 安装依赖
log_step "步骤 3: 安装依赖包"

# 如果环境已存在，询问是否需要安装/更新依赖
INSTALL_DEPS=true
if [ "$ENV_EXISTS" = "true" ]; then
    echo ""
    echo "=========================================="
    read -p ">>> 是否需要安装/更新依赖包? (Y/n): " -r response </dev/tty
    echo "=========================================="
    echo ""
    if [[ "$response" =~ ^[Nn]$ ]]; then
        INSTALL_DEPS=false
    fi
fi

if [ "$INSTALL_DEPS" = "true" ]; then
    log_info "安装依赖包..."
    pip install --upgrade pip setuptools wheel -q
    pip install pytest pytest-randomly pytest-anyio -q

    # 检查是否有 requirements.txt
    if [ -f "$PROJECT_DIR/requirements.txt" ]; then
        TEMP_REQ=$(mktemp)
        grep -v "^mindspore" "$PROJECT_DIR/requirements.txt" | \
        grep -v "^torch" | \
        grep -v "^#.*torch" > "$TEMP_REQ" || true
        
        if [ -s "$TEMP_REQ" ]; then
            pip install -r "$TEMP_REQ" -q || {
                log_warn "部分依赖安装失败，继续..."
            }
        fi
        rm -f "$TEMP_REQ"
    else
        pip install numpy -q
    fi

    # 验证 pytest
    python -c "import pytest" 2>/dev/null || {
        pip install pytest pytest-randomly pytest-anyio --force-reinstall -q
    }

    log_info "✅ 依赖安装完成"
fi

# 步骤 4: 安装 PyTorch 2.1.0
log_step "步骤 4: 安装 PyTorch 2.1.0"
echo ""

# 如果环境已存在，询问是否需要安装/更新 PyTorch
INSTALL_PYTORCH=true
if [ "$ENV_EXISTS" = "true" ]; then
    if python -c "import torch" 2>/dev/null; then
        INSTALLED_TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        log_warn "PyTorch 已安装，版本: $INSTALLED_TORCH_VERSION"
        echo ""
        echo "=========================================="
        read -p ">>> 是否需要安装/更新 PyTorch 2.1.0? (Y/n): " -r response </dev/tty
        echo "=========================================="
        echo ""
        if [[ "$response" =~ ^[Nn]$ ]]; then
            INSTALL_PYTORCH=false
            log_info "✓ 保持现有 PyTorch 版本: $INSTALLED_TORCH_VERSION"
        else
            log_info "✓ 将安装/更新 PyTorch 2.1.0"
        fi
    else
        log_warn "PyTorch 未安装"
        echo ""
        echo "=========================================="
        read -p ">>> 是否需要安装 PyTorch 2.1.0? (Y/n): " -r response </dev/tty
        echo "=========================================="
        echo ""
        if [[ "$response" =~ ^[Nn]$ ]]; then
            INSTALL_PYTORCH=false
            log_info "✓ 跳过 PyTorch 安装"
        else
            log_info "✓ 将安装 PyTorch 2.1.0"
        fi
    fi
fi
echo ""

if [ "$INSTALL_PYTORCH" = "true" ]; then
    PYTORCH_VERSION="2.1.0"
    if python -c "import torch" 2>/dev/null; then
        INSTALLED_TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        BASE_VERSION=$(echo "$INSTALLED_TORCH_VERSION" | cut -d'+' -f1)
        if [ "$BASE_VERSION" != "$PYTORCH_VERSION" ]; then
            log_info "安装 PyTorch ${PYTORCH_VERSION}..."
            pip install --force-reinstall "torch==${PYTORCH_VERSION}" -q || {
                log_warn "PyTorch 安装失败，继续使用现有版本"
            }
        fi
    else
        log_info "安装 PyTorch ${PYTORCH_VERSION}..."
        pip install "torch==${PYTORCH_VERSION}" -q || {
            log_warn "PyTorch 安装失败，但可以继续"
        }
    fi
    log_info "✅ PyTorch 安装完成"
fi

# 步骤 5: 安装 MindSpore
log_step "步骤 5: 安装 MindSpore"
echo ""

# 如果环境已存在，询问是否需要更新/安装 MindSpore
UPDATE_MINDSPORE=true
if [ "$ENV_EXISTS" = "true" ]; then
    if python -c "import mindspore" 2>/dev/null; then
        INSTALLED_MS_VERSION=$(python -c "import mindspore; print(mindspore.__version__)" 2>/dev/null)
        echo ""
        log_warn "MindSpore 已安装，版本: $INSTALLED_MS_VERSION"
        echo ""
        echo "=========================================="
        read -p ">>> 是否需要更新 MindSpore 版本? (Y/n): " -r response </dev/tty
        echo "=========================================="
        echo ""
        if [[ "$response" =~ ^[Nn]$ ]]; then
            UPDATE_MINDSPORE=false
            log_info "✓ 保持现有 MindSpore 版本: $INSTALLED_MS_VERSION"
        else
            log_info "✓ 将更新 MindSpore 版本"
        fi
    else
        echo ""
        log_warn "MindSpore 未安装"
        echo ""
        echo "=========================================="
        read -p ">>> 是否需要安装 MindSpore? (Y/n): " -r response </dev/tty
        echo "=========================================="
        echo ""
        if [[ "$response" =~ ^[Nn]$ ]]; then
            UPDATE_MINDSPORE=false
            log_info "✓ 跳过 MindSpore 安装"
        else
            log_info "✓ 将安装 MindSpore"
        fi
    fi
fi
echo ""

if [ "$UPDATE_MINDSPORE" = "true" ]; then
    # 确定 Python 版本（用于查找对应的 whl 文件）
    PYTHON_MAJOR_MINOR=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null || echo "39")

    # 判断是否使用 daily 版本
    if [ "$MINDSPORE_VERSION" = "daily" ] || [ "$USE_DAILY" = "true" ]; then
        log_info "使用 MindSpore daily 版本..."
        
        # 获取最新的 daily 版本
        if get_latest_mindspore_daily "$PYTHON_MAJOR_MINOR"; then
            log_info "下载 MindSpore daily 版本: $MINDSPORE_DAILY_VERSION"
            
            # 下载到临时目录，使用原始文件名（pip 需要正确的 wheel 文件名格式）
            ORIGINAL_WHL_NAME="$MINDSPORE_DAILY_FILE"
            TEMP_DIR=$(mktemp -d)
            TEMP_WHL="${TEMP_DIR}/${ORIGINAL_WHL_NAME}"
            download_url="$MINDSPORE_DAILY_URL"
            
            log_info "下载 MindSpore daily 版本（文件较大，请耐心等待）..."
            # 使用 curl 下载，检查退出码和文件大小
            curl -L -f --progress-bar "$download_url" -o "$TEMP_WHL" 2>&1
            curl_exit_code=$?
            
            # 检查下载是否成功
            if [ $curl_exit_code -eq 0 ] && [ -f "$TEMP_WHL" ] && [ -s "$TEMP_WHL" ]; then
                # 检查文件大小（whl 文件通常 > 100MB）
                file_size=$(stat -f%z "$TEMP_WHL" 2>/dev/null || stat -c%s "$TEMP_WHL" 2>/dev/null || echo "0")
                if [ "$file_size" -gt 100000000 ]; then  # 大于 100MB
                    log_info "✅ 下载完成 ($(numfmt --to=iec-i --suffix=B $file_size 2>/dev/null || echo "${file_size} bytes"))"
                    log_info "安装 MindSpore daily 版本..."
                    pip install --force-reinstall "$TEMP_WHL" -q || {
                        log_error "MindSpore 安装失败"
                        rm -rf "$TEMP_DIR"
                        exit 1
                    }
                    rm -rf "$TEMP_DIR"
                else
                    log_error "下载的文件大小异常（${file_size} bytes），可能下载失败"
                    rm -rf "$TEMP_DIR"
                    # 回退到稳定版本
                    FALLBACK_VERSION="2.7.2"
                    log_info "回退到稳定版本: ${FALLBACK_VERSION}"
                    pip install --force-reinstall "mindspore==${FALLBACK_VERSION}" || {
                        log_error "MindSpore 安装失败"
                        log_info "提示: 可以手动下载 whl 文件: $download_url"
                        exit 1
                    }
                fi
        else
            log_error "下载失败，回退到稳定版本 2.7.2"
            rm -rf "$TEMP_DIR"
            pip install --force-reinstall "mindspore==2.7.2" -q || {
                log_error "MindSpore 安装失败"
                exit 1
            }
        fi
        else
            log_warn "获取 daily 版本失败，使用稳定版本"
            if [ "$MINDSPORE_VERSION" = "daily" ]; then
                MINDSPORE_VERSION="2.7.2"
            fi
            pip install --force-reinstall "mindspore==${MINDSPORE_VERSION}" -q || {
                log_error "MindSpore 安装失败"
                exit 1
            }
        fi
    else
        log_info "安装 MindSpore ${MINDSPORE_VERSION}..."
        pip install --force-reinstall "mindspore==${MINDSPORE_VERSION}" -q || {
            log_error "MindSpore 安装失败"
            exit 1
        }
    fi
    log_info "✅ MindSpore 安装完成"
    
    # 如果更新了 MindSpore，自动设置需要重新编译 op-plugin 以确保兼容性
    if [ "$ENV_EXISTS" = "true" ]; then
        if [ -f "$PROJECT_DIR/build/ms_op_plugin/lib/libms_op_plugin.dylib" ] || \
           [ -f "$PROJECT_DIR/build/libms_op_plugin.dylib" ]; then
            SKIP_BUILD="false"  # 强制重新编译
        fi
    fi
fi

# 步骤 6: 检查 libtorch 库文件
log_step "步骤 6: 检查 libtorch 库文件"

cd "$PROJECT_DIR"

# libtorch 库文件路径（已合入代码仓）
LIBTORCH_DIR="third_party/libtorch/lib/x86_64/darwin"

# 检查 libtorch 是否存在
if [ ! -d "$LIBTORCH_DIR" ] || [ -z "$(ls -A "$LIBTORCH_DIR"/*.dylib 2>/dev/null)" ]; then
    log_error "❌ 未找到 libtorch 库文件: $LIBTORCH_DIR"
    log_error "请确保 libtorch 已合入代码仓"
    log_info "预期路径: $PROJECT_DIR/$LIBTORCH_DIR"
    exit 1
fi

# 验证必需的库文件
if [ ! -f "$LIBTORCH_DIR/libc10.dylib" ] || [ ! -f "$LIBTORCH_DIR/libtorch_cpu.dylib" ]; then
    log_error "❌ 缺少必需的 libtorch 库文件"
    log_info "需要文件:"
    log_info "  - $LIBTORCH_DIR/libc10.dylib"
    log_info "  - $LIBTORCH_DIR/libtorch_cpu.dylib"
    exit 1
fi

log_info "✅ libtorch 库文件已就绪"

# 步骤 7: 编译 op-plugin
log_step "步骤 7: 编译 op-plugin"

# 如果环境已存在且未更新 MindSpore，询问是否需要重新编译
# 如果更新了 MindSpore，已在步骤 5 自动设置 SKIP_BUILD=false，直接编译
if [ "$ENV_EXISTS" = "true" ] && [ "$SKIP_BUILD" != "true" ] && [ "$UPDATE_MINDSPORE" != "true" ]; then
    if [ -f "$PROJECT_DIR/build/ms_op_plugin/lib/libms_op_plugin.dylib" ] || \
       [ -f "$PROJECT_DIR/build/libms_op_plugin.dylib" ]; then
        echo ""
        log_warn "op-plugin 已编译"
        echo ""
        echo "=========================================="
        read -p ">>> 是否需要重新编译 op-plugin? (Y/n): " -r response </dev/tty
        echo "=========================================="
        echo ""
        if [[ "$response" =~ ^[Nn]$ ]]; then
            SKIP_BUILD="true"
            log_info "✓ 跳过编译，使用已编译的 op-plugin"
        else
            log_info "✓ 将重新编译 op-plugin"
        fi
    fi
fi

if [ "$SKIP_BUILD" != "true" ]; then
    
    if ! command -v cmake &> /dev/null; then
        log_info "安装 CMake..."
        conda install cmake -y -q || {
            log_error "CMake 安装失败，请手动安装: brew install cmake"
            exit 1
        }
    fi
    
    log_info "开始编译 op-plugin..."
    if [ ! -f "$PROJECT_DIR/build.sh" ]; then
        log_error "未找到 build.sh 文件: $PROJECT_DIR/build.sh"
        exit 1
    fi
    bash "$PROJECT_DIR/build.sh" || {
        log_error "编译失败"
        exit 1
    }
    
    # 验证编译结果
    if [ -f "build/libms_op_plugin.dylib" ] || [ -f "build/ms_op_plugin/lib/libms_op_plugin.dylib" ]; then
        log_info "✅ 编译成功"
    else
        log_error "❌ 未找到编译生成的库文件"
        exit 1
    fi
else
    log_info "跳过编译（SKIP_BUILD=true）"
fi

# 步骤 8: 设置环境变量
log_step "步骤 8: 设置环境变量"

if [ ! -f "$PROJECT_DIR/env.source" ]; then
    log_error "未找到 env.source 文件: $PROJECT_DIR/env.source"
    exit 1
fi
source "$PROJECT_DIR/env.source"

# 步骤 9: 验证 op-plugin 加载
log_step "步骤 9: 验证 op-plugin 加载"

if python -c "import ms_op_plugin" 2>/dev/null; then
    log_info "✅ op-plugin 加载成功"
else
    ERROR_OUTPUT=$(python -c "import ms_op_plugin" 2>&1 || true)
    if echo "$ERROR_OUTPUT" | grep -q "incompatible architecture"; then
        log_error "❌ 架构不匹配错误"
        exit 1
    else
        log_warn "⚠️  op-plugin 导入有警告"
    fi
fi

echo ""
echo "=========================================="
echo "  ✅ 完成！"
echo "=========================================="
echo ""
log_info "后续使用说明:"
log_info "  1. conda activate $ENV_NAME"
log_info "  2. cd $PROJECT_DIR"
log_info "  3. source env.source"
echo ""
