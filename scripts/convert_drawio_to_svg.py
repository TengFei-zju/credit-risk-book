#!/usr/bin/env python3
"""
Convert drawio files to SVG format for GitHub rendering.

Usage:
    python scripts/convert_drawio_to_svg.py

Requirements:
    pip install drawio-export

Or manually:
    1. Open draw.io desktop app
    2. File > Export as > SVG
    3. Save to the same directory as the .drawio file
"""

import os
import re
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "chapters" / "diagrams"


def convert_drawio_to_svg_manually():
    """
    由于 drawio 转换需要图形界面环境，
    这里提供手动导出步骤的指引。
    """
    drawio_files = list(DIAGRAMS_DIR.glob("*.drawio"))

    print("=" * 60)
    print("Drawio to SVG 转换指引")
    print("=" * 60)
    print(f"\n找到 {len(drawio_files)} 个 drawio 文件:\n")

    for f in drawio_files:
        print(f"  - {f.name}")

    print("\n" + "=" * 60)
    print("手动导出步骤:")
    print("=" * 60)
    print("""
1. 打开 draw.io 桌面应用 (或访问 app.diagrams.net)

2. 打开每个 .drawio 文件

3. 点击菜单：File > Export as > SVG

4. 保存设置:
   - 勾选 "Embed Images" (嵌入图片)
   - 勾选 "Replace existing file" (如果存在同名文件)
   - 保存为同名 .svg 文件（与原 .drawio 文件同目录）

5. 或者使用批处理（如果安装了 draw.io CLI）:
   draw.io -x -f svg -o output.svg input.drawio
    """)

    print("=" * 60)
    print("或者，使用以下 Python 库（需要图形环境）:")
    print("=" * 60)
    print("""
pip install drawio-export

然后在有图形界面的环境下运行:
    python scripts/convert_drawio_to_svg.py
    """)


def update_markdown_references():
    """
    更新 Markdown 文件中的 drawio 引用为 svg 引用。
    """
    markdown_files = list((PROJECT_ROOT / "chapters").glob("*.md"))

    changes = []
    for md_file in markdown_files:
        content = md_file.read_text(encoding='utf-8')
        original = content

        # 替换 .drawio 为 .svg
        content = re.sub(r'\.drawio\)', '.svg)', content)

        if content != original:
            md_file.write_text(content, encoding='utf-8')
            changes.append(md_file.name)

    return changes


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--update-md":
        # 仅更新 Markdown 引用
        changed = update_markdown_references()
        print(f"已更新以下文件的引用：{changed}")
    else:
        # 显示转换指引
        convert_drawio_to_svg_manually()

        # 先更新 Markdown 引用（即使 SVG 还未生成）
        print("\n正在更新 Markdown 引用...")
        changed = update_markdown_references()
        print(f"已更新：{changed}")
        print("\n下一步：请使用 draw.io 导出 SVG 文件到 diagrams/ 目录")
