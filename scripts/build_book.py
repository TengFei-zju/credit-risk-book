#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图书构建脚本
每次 commit 时自动执行：
1. 更新版本号
2. 生成 PDF
"""

import sys
import subprocess
from pathlib import Path


def run_script(script_name, description=""):
    """运行脚本"""
    if description:
        print(f"\n{'='*60}")
        print(f"  {description}")
        print(f"{'='*60}\n")

    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"  错误：脚本不存在 {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  错误：{e}")
        return False


def add_pdf_to_git():
    """将生成的 PDF 添加到 git 暂存区"""
    print("\n  将 PDF 添加到 git...")

    root_dir = Path(__file__).parent.parent
    pdf_file = root_dir / "信贷风控建模手册.pdf"

    if pdf_file.exists():
        try:
            subprocess.run(
                ["git", "add", str(pdf_file)],
                cwd=str(root_dir),
                capture_output=True
            )
            print(f"  已添加：信贷风控建模手册.pdf")
            return True
        except Exception as e:
            print(f"  警告：添加 PDF 到 git 失败 - {e}")
            return False
    else:
        print("  警告：PDF 文件未生成")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  信贷风控建模：打工人手册 - 自动构建")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent
    scripts_dir = root_dir / "scripts"

    # 1. 更新版本号
    success_version = run_script(
        "update_version.py",
        "步骤 1: 更新版本号"
    )

    # 2. 生成 PDF
    success_pdf = run_script(
        "generate_pdf.py",
        "步骤 2: 生成 PDF"
    )

    # 3. 添加 PDF 到 git（如果生成了 PDF）
    if success_pdf:
        add_pdf_to_git()

    # 总结
    print("\n" + "=" * 60)
    print("  构建完成!")
    print("=" * 60)

    if success_version and success_pdf:
        print("\n  所有步骤成功完成")
        return 0
    else:
        print("\n  部分步骤失败，请检查输出信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
