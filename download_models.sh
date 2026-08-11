#!/usr/bin/env bash
#
# Vocal2Midi — 模型下载/链接脚本
# ================================
# 用法:
#   bash download_models.sh                   # 全部从网络下载
#   bash download_models.sh --from-openutau   # 优先复用 OpenUTAU 已有模型
#
# 模型清单:
#   1. GAME        (音高/音符提取)     ~172 MB  或 OpenUTAU 复用
#   2. HubertFA    (强制对齐)          ~245 MB  须下载
#   3. Qwen3-ASR   (语音识别, int4)   ~2.7 GB  须下载
#   4. RMVPE       (音高曲线)          ~362 MB  或 OpenUTAU 复用
#   5. llama.cpp   (Qwen 解码器)      需编译
#   6. RomajiASR   (日语识别)          ~168 MB  须下载
# ================================

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
EXPERIMENTS="$ROOT/experiments"
OPENUTAU="${OPENUTAU_DIR:-$HOME/.local/share/OpenUtau}"
USE_OPENUTAU=false

[[ "${1:-}" == "--from-openutau" ]] && USE_OPENUTAU=true

mkdir -p "$EXPERIMENTS"

info()  { echo -e "\033[34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[32m[ OK ]\033[0m $*"; }
warn()  { echo -e "\033[33m[WARN]\033[0m $*"; }
err()   { echo -e "\033[31m[ERR ]\033[0m $*"; }

HAVE_WGET=false; command -v wget >/dev/null 2>&1 && HAVE_WGET=true
HAVE_CURL=false; command -v curl >/dev/null 2>&1 && HAVE_CURL=true

download() {
    local url="$1" out="$2" label="$3"
    if [ -f "$out" ]; then
        ok "$label 已存在，跳过"; return 0
    fi
    info "下载 $label ..."
    mkdir -p "$(dirname "$out")"
    if $HAVE_WGET; then wget -q --show-progress -O "$out" "$url"
    elif $HAVE_CURL; then curl -#SL -o "$out" "$url"
    else err "需要 wget 或 curl"; return 1; fi
    ok "$label 下载完成"
}

# 检查目标目录是否包含有效的模型文件（排除单层嵌套目录）
_dir_has_model_files() {
    local dir="$1"
    local marker="${2:-}"  # 可选：标志文件路径（相对 dest）
    if [ ! -d "$dir" ]; then return 1; fi
    # 如果有指定的标志文件（相对于 dir），直接检查
    if [ -n "$marker" ]; then
        [ -f "$dir/$marker" ] && return 0
        return 1
    fi
    # 否则检查是否直接包含 .onnx 文件
    local count; count="$(find "$dir" -maxdepth 1 -name '*.onnx' -type f 2>/dev/null | wc -l)"
    [ "$count" -gt 0 ]
}

# 展平：如果解压后只有一个子目录，把内容提上来
_flatten_single_subdir() {
    local dir="$1"
    local entries
    entries=("$dir"/*)
    if [ ${#entries[@]} -eq 1 ] && [ -d "${entries[0]}" ]; then
        info "检测到嵌套目录，展平: $(basename "${entries[0]}")"
        # 用临时目录做交换
        local tmpd; tmpd="$(mktemp -d)"
        mv "$dir"/*/* "$tmpd/" 2>/dev/null || true
        # 有可能有隐藏文件
        shopt -s dotglob 2>/dev/null || true
        mv "$dir"/*/* "$tmpd/" 2>/dev/null || true
        shopt -u dotglob 2>/dev/null || true
        rm -rf "${entries[0]}"
        mv "$tmpd"/* "$dir/" 2>/dev/null || true
        mv "$tmpd"/.[!.]* "$dir/" 2>/dev/null || true
        rmdir "$tmpd" 2>/dev/null || true
    fi
}

download_unzip() {
    local url="$1" dest="$2" label="$3" marker="${4:-model.onnx}" tmpf
    if _dir_has_model_files "$dest" "$marker"; then
        ok "$label 已存在，跳过"; return 0
    fi
    info "下载 $label ..."
    mkdir -p "$dest"
    tmpf="$(mktemp)"
    if $HAVE_WGET; then wget -q --show-progress -O "$tmpf" "$url"
    elif $HAVE_CURL; then curl -#SL -o "$tmpf" "$url"
    else err "需要 wget 或 curl"; return 1; fi
    info "解压 $label ..."
    unzip -q -o "$tmpf" -d "$dest"
    rm -f "$tmpf"
    _flatten_single_subdir "$dest"
    ok "$label 下载并解压完成"
}

download_targz() {
    local url="$1" dest="$2" label="$3" marker="${4:-config.json}" tmpf
    if _dir_has_model_files "$dest" "$marker"; then
        ok "$label 已存在，跳过"; return 0
    fi
    info "下载 $label (较大文件，请耐心等待) ..."
    mkdir -p "$dest"
    tmpf="$(mktemp)"
    if $HAVE_WGET; then wget -q --show-progress -O "$tmpf" "$url"
    elif $HAVE_CURL; then curl -#SL -o "$tmpf" "$url"
    else err "需要 wget 或 curl"; return 1; fi
    info "解压 $label ..."
    tar xzf "$tmpf" -C "$dest"
    rm -f "$tmpf"
    _flatten_single_subdir "$dest"
    ok "$label 下载并解压完成"
}

symlink_dir() {
    local src="$1" link="$2" label="$3"
    if [ ! -d "$src" ]; then
        warn "OpenUTAU 中未找到 $label: $src"
        return 1
    fi
    if [ -d "$link" ] || [ -L "$link" ]; then
        ok "$label 已存在，跳过"
        return 0
    fi
    ln -sfn "$src" "$link"
    ok "$label → 已链接到 OpenUTAU ($src)"
}

symlink_file() {
    local src="$1" link="$2" label="$3"
    if [ ! -f "$src" ]; then
        warn "OpenUTAU 中未找到 $label: $src"
        return 1
    fi
    if [ -f "$link" ] || [ -L "$link" ]; then
        ok "$label 已存在，跳过"
        return 0
    fi
    mkdir -p "$(dirname "$link")"
    ln -sfn "$src" "$link"
    ok "$label → 已链接到 OpenUTAU ($src)"
}

# ================================================================
# 1. GAME — 音高/音符提取
# ================================================================
echo ""
info "========== 1/6: GAME (音高/音符提取) =========="
if $USE_OPENUTAU && symlink_dir \
    "$OPENUTAU/Dependencies/game" \
    "$EXPERIMENTS/GAME-1.0.3-medium-onnx" \
    "GAME"; then
    :  # linked
else
    download_unzip \
        "https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-medium-onnx.zip" \
        "$EXPERIMENTS/GAME-1.0.3-medium-onnx" \
        "GAME-1.0.3-medium-onnx" \
        "encoder.onnx"
fi

# ================================================================
# 2. HubertFA — 强制对齐
# ================================================================
echo ""
info "========== 2/6: HubertFA (强制对齐) =========="
download_unzip \
    "https://github.com/wolfgitpr/HubertFA/releases/download/v0.0.7/1218_hfa_model_new_dict.zip" \
    "$EXPERIMENTS/1218_hfa_model_new_dict" \
    "1218_hfa_model_new_dict" \
    "model.onnx"

# ================================================================
# 3. Qwen3-ASR — 语音识别 (推荐 int4 量化版)
# ================================================================
echo ""
info "========== 3/6: Qwen3-ASR (语音识别 int4) =========="
download_targz \
    "https://huggingface.co/andrewleech/qwen3-asr-1.7b-onnx/resolve/main/qwen3-asr-1.7b-int4.tar.gz" \
    "$EXPERIMENTS/Qwen3-ASR-1.7B-dml" \
    "Qwen3-ASR-1.7B-int4" \
    "config.json"

# ================================================================
# 4. RMVPE — 音高曲线
# ================================================================
echo ""
info "========== 4/6: RMVPE (音高曲线) =========="
RMVPE_DIR="$EXPERIMENTS/RMVPE"
mkdir -p "$RMVPE_DIR"
if $USE_OPENUTAU && symlink_file \
    "$OPENUTAU/Dependencies/rmvpe/rmvpe.onnx" \
    "$RMVPE_DIR/rmvpe.onnx" \
    "RMVPE"; then
    :  # linked
else
    download \
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx" \
        "$RMVPE_DIR/rmvpe.onnx" \
        "RMVPE/rmvpe.onnx"
fi

# ================================================================
# 5. llama.cpp — Qwen 解码器共享库
# ================================================================
echo ""
info "========== 5/6: llama.cpp 共享库 (Qwen 解码器) =========="
LLAMA_BIN="$ROOT/inference/qwen3asr_dml/bin"

case "$(uname -s)" in
    Darwin)
        LLAMA_EXT="dylib"
        BUILD_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 1)"
        ;;
    *)
        LLAMA_EXT="so"
        BUILD_JOBS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
        ;;
esac

_libllama_exists() {
    if [ ! -f "$LLAMA_BIN/libllama.$LLAMA_EXT" ] \
        || [ ! -f "$LLAMA_BIN/libggml.$LLAMA_EXT" ] \
        || [ ! -f "$LLAMA_BIN/libggml-base.$LLAMA_EXT" ]; then
        return 1
    fi
    if [ "$LLAMA_EXT" = "dylib" ]; then
        # llama.cpp's macOS shared build links these backend libraries through
        # @rpath, so they must travel with the three core libraries.
        for required in libggml-cpu.dylib libggml-blas.dylib libggml-metal.dylib; do
            [ -f "$LLAMA_BIN/$required" ] || return 1
        done
    fi
    return 0
}

_resolve_libs_from_build() {
    local build_dir="$1"
    local ret=1  # 默认失败

    # libllama shared library
    local llama_src
    llama_src="$(find "$build_dir" -name "libllama.$LLAMA_EXT" -type f 2>/dev/null | head -1)"
    if [ -n "$llama_src" ]; then
        mkdir -p "$LLAMA_BIN"
        cp -L "$llama_src" "$LLAMA_BIN/libllama.$LLAMA_EXT"
    fi

    # ggml — 优先 libggml，后备 libggml_shared
    local ggml_src
    ggml_src="$(find "$build_dir" -name "libggml.$LLAMA_EXT" -type f 2>/dev/null | head -1)"
    if [ -z "$ggml_src" ]; then
        ggml_src="$(find "$build_dir" -name "libggml_shared.$LLAMA_EXT" -type f 2>/dev/null | head -1)"
    fi
    if [ -n "$ggml_src" ]; then
        cp -L "$ggml_src" "$LLAMA_BIN/libggml.$LLAMA_EXT"
    fi

    # ggml-base — 优先 libggml-base，后备 ggml 或 ggml_shared
    local ggml_base_src
    ggml_base_src="$(find "$build_dir" -name "libggml-base.$LLAMA_EXT" -type f 2>/dev/null | head -1)"
    if [ -z "$ggml_base_src" ]; then
        # 如果已经复制了 libggml，就用它当后备库
        if [ -f "$LLAMA_BIN/libggml.$LLAMA_EXT" ]; then
            cp -L "$LLAMA_BIN/libggml.$LLAMA_EXT" "$LLAMA_BIN/libggml-base.$LLAMA_EXT"
            ggml_base_src="alias"
        fi
    else
        cp -L "$ggml_base_src" "$LLAMA_BIN/libggml-base.$LLAMA_EXT"
    fi

    # Copy backend libraries (CPU, BLAS, Metal, etc.) alongside the core
    # libraries. They are separate shared objects in the llama.cpp build.
    while IFS= read -r backend_src; do
        [ -n "$backend_src" ] || continue
        cp -L "$backend_src" "$LLAMA_BIN/$(basename "$backend_src")"
    done < <(
        find "$build_dir" -type f \( \
            -name "libggml*.$LLAMA_EXT" -o \
            -name "ggml*.$LLAMA_EXT" \
        \) 2>/dev/null
    )

    # macOS llama.cpp builds usually embed Metal in libggml. Copy an optional
    # standalone backend too, because older revisions may load it dynamically.
    if [ "$LLAMA_EXT" = "dylib" ]; then
        while IFS= read -r metal_src; do
            [ -n "$metal_src" ] || continue
            cp -L "$metal_src" "$LLAMA_BIN/$(basename "$metal_src")"
        done < <(
            find "$build_dir" -type f \( \
                -name 'libggml-metal*.dylib' -o \
                -name 'libggml-metal*.so' -o \
                -name 'ggml-metal*.dylib' -o \
                -name 'ggml-metal*.so' \
            \) 2>/dev/null
        )
    fi

    # 检查结果
    if _libllama_exists; then
        ret=0
    fi
    return $ret
}

_copy_system_libs() {
    local sys_llama="" sys_ggml="" found root
    for root in /usr/lib /usr/local/lib /opt/homebrew/lib /opt/local/lib "$HOME/.local/lib"; do
        [ -d "$root" ] || continue
        if [ -z "$sys_llama" ]; then
            sys_llama="$(find "$root" -name "libllama.$LLAMA_EXT" -type f 2>/dev/null | head -1)"
        fi
        found="$(find "$root" -name "libggml*.$LLAMA_EXT" -type f 2>/dev/null || true)"
        if [ -n "$found" ]; then
            sys_ggml="${sys_ggml}${sys_ggml:+$'\n'}$found"
        fi
    done
    if [ -z "$sys_llama" ] || [ -z "$sys_ggml" ]; then
        return 1
    fi
    info "发现系统安装的 llama.cpp，复制共享库..."
    mkdir -p "$LLAMA_BIN"
    cp -L "$sys_llama" "$LLAMA_BIN/libllama.$LLAMA_EXT"
    local have_ggml=false have_ggml_base=false
    for f in $sys_ggml; do
        case "$(basename "$f")" in
            "libggml-base.$LLAMA_EXT") cp -L "$f" "$LLAMA_BIN/libggml-base.$LLAMA_EXT"; have_ggml_base=true ;;
            "libggml.$LLAMA_EXT")      cp -L "$f" "$LLAMA_BIN/libggml.$LLAMA_EXT";      have_ggml=true ;;
            libggml-metal*.$LLAMA_EXT|ggml-metal*.$LLAMA_EXT)
                cp -L "$f" "$LLAMA_BIN/$(basename "$f")" ;;
            "libggml_shared.$LLAMA_EXT")
                if ! $have_ggml; then cp -L "$f" "$LLAMA_BIN/libggml.$LLAMA_EXT"; have_ggml=true; fi
                if ! $have_ggml_base; then cp -L "$f" "$LLAMA_BIN/libggml-base.$LLAMA_EXT"; have_ggml_base=true; fi
                ;;
            libggml*.$LLAMA_EXT|ggml*.$LLAMA_EXT)
                cp -L "$f" "$LLAMA_BIN/$(basename "$f")" ;;
        esac
    done
    if _libllama_exists; then
        return 0
    fi
    return 1
}

# ── llama.cpp 主逻辑 ──
if _libllama_exists; then
    ok "llama.cpp 共享库已存在，跳过"
else
    _copy_system_libs && ok "已从系统复制 llama.cpp 共享库"

    if ! _libllama_exists; then
        info "从源码编译 llama.cpp 共享库..."
        BUILD_DIR="$(mktemp -d)"
        cd "$BUILD_DIR"

        # Keep this pinned to the API generation used by inference/qwen3asr_dml/llama.py.
        LLAMA_TAG="b9000"
        if ! git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp.git 2>/dev/null; then
            err "克隆兼容版本 llama.cpp $LLAMA_TAG 失败"
            err "请检查网络，或手动将兼容版本的共享库复制到 $LLAMA_BIN/"
            cd "$ROOT"
            rm -rf "$BUILD_DIR"
            exit 1
        fi
        cd llama.cpp
        CMAKE_ARGS=(
            -DBUILD_SHARED_LIBS=ON \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_TOOLS=OFF \
            -DLLAMA_BUILD_COMMON=OFF \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_SERVER=OFF \
            -DCMAKE_BUILD_TYPE=Release
        )
        if [ "$LLAMA_EXT" = "dylib" ]; then
            # Apple Silicon uses Metal for llama.cpp decoder offload. Keep it
            # embedded so the runtime works without a separately named plugin.
            CMAKE_ARGS+=(
                -DGGML_METAL=ON
                -DGGML_METAL_EMBED_LIBRARY=ON
                -DCMAKE_BUILD_RPATH=@loader_path
                -DCMAKE_INSTALL_RPATH=@loader_path
                -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
            )
        fi
        cmake -B build "${CMAKE_ARGS[@]}"
        cmake --build build -j"$BUILD_JOBS"

        mkdir -p "$LLAMA_BIN"
        _resolve_libs_from_build "build"

        cd "$ROOT"
        rm -rf "$BUILD_DIR"

        if _libllama_exists; then
            ok "llama.cpp 编译完成，共享库已复制到 $LLAMA_BIN"
        else
            err "llama.cpp 编译完成但未能定位共享库文件。"
            err "编译产出可能在 /tmp 下，手动查找:"
            err "  find /tmp -name 'libllama.*' -type f"
            err "  find /tmp -name 'libggml*' -type f"
        fi
    fi
fi

# ================================================================
# 6. RomajiASR — 日语语音识别
# ================================================================
echo ""
info "========== 6/6: RomajiASR (日语识别) =========="
download_unzip \
    "https://github.com/Xiantaidu/RomajiASR/releases/download/v1.0.0/model_onnx.zip" \
    "$EXPERIMENTS/romajiASR" \
    "RomajiASR" \
    "model.onnx"

# ================================================================
echo ""
echo "=============================================="
ok "所有模型处理完成！"
echo ""
echo "模型目录: $EXPERIMENTS"
echo ""
echo "模型清单:"
for d in "$EXPERIMENTS"/*/; do
    name="$(basename "$d")"
    if [ -L "$d" ]; then
        echo "  📎 $name → $(readlink "$d")"
    else
        echo "  📁 $name"
    fi
done
for f in "$EXPERIMENTS"/*/*.onnx; do
    [ -f "$f" ] || continue
    if [ -L "$f" ]; then
        echo "  📎 $(basename "$(dirname "$f")")/$(basename "$f") → $(readlink "$f")"
    fi
done
echo ""
echo "下一步:"
echo "  启动 GUI:  uv run vocal2midi"
echo "  或 CLI:    uv run slice-asr --help"
echo "=============================================="
