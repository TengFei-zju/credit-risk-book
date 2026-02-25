#!/bin/bash
# Draw.io 批量导出 SVG 脚本
# 使用方法（在安装了 draw.io 的 Windows 系统上）:
#   bash scripts/export_drawio_to_svg.sh

DIAGRAMS_DIR="chapters/diagrams"

# draw.io 命令行工具路径（Windows）
DRAWIO_PATH="/c/Program Files/draw.io/draw.io.exe"

echo "=============================================="
echo "Draw.io 批量导出 SVG"
echo "=============================================="

# 检查 draw.io 是否安装
if [ ! -f "$DRAWIO_PATH" ]; then
    echo "未找到 draw.io，尝试使用备用路径..."
    DRAWIO_PATH="/c/Program Files (x86)/draw.io/draw.io.exe"
fi

if [ ! -f "$DRAWIO_PATH" ]; then
    echo "错误：未找到 draw.io 安装"
    echo ""
    echo "请手动导出 SVG："
    echo "1. 打开 draw.io 桌面应用或访问 app.diagrams.net"
    echo "2. 打开每个 .drawio 文件"
    echo "3. File > Export as > SVG"
    echo "4. 保存到 diagrams/ 目录"
    exit 1
fi

echo "使用 draw.io: $DRAWIO_PATH"
echo ""

# 导出每个 drawio 文件为 SVG
for drawio_file in "$DIAGRAMS_DIR"/*.drawio; do
    if [ -f "$drawio_file" ]; then
        svg_file="${drawio_file%.drawio}.svg"
        echo "导出：$drawio_file -> $svg_file"
        "$DRAWIO_PATH" -x -f svg -o "$svg_file" "$drawio_file"
    fi
done

echo ""
echo "导出完成！"
