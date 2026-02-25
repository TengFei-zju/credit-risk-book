#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
版本号自动更新脚本
更新 pyproject.toml 和 README.md 中的版本号
"""

import re
from pathlib import Path
from datetime import datetime


def get_current_version(pyproject_path):
    """从 pyproject.toml 获取当前版本号"""
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        content = f.read()

    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    return "0.0.0"


def parse_version(version_str):
    """解析版本号为主版本.次版本.修订号"""
    parts = version_str.split('.')
    if len(parts) == 2:
        return int(parts[0]), int(parts[1]), 0
    elif len(parts) >= 3:
        return int(parts[0]), int(parts[1]), int(parts[2])
    else:
        return 0, 0, 0


def increment_version(version_str, increment_type='patch'):
    """递增版本号"""
    major, minor, patch = parse_version(version_str)

    if increment_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif increment_type == 'minor':
        minor += 1
        patch = 0
    else:  # patch
        patch += 1

    return f"{major}.{minor}.{patch}"


def update_pyproject_version(pyproject_path, new_version):
    """更新 pyproject.toml 中的版本号"""
    with open(pyproject_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换版本号
    new_content = re.sub(
        r'(version\s*=\s*)"[^"]+"',
        rf'\1"{new_version}"',
        content
    )

    with open(pyproject_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True


def update_readme_version(readme_path, new_version):
    """更新 README.md 中的版本号和更新日期"""
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 更新版本号 (格式：版本：v0.2)
    new_content = re.sub(
        r'版本：v[\d.]+',
        f'版本：v{new_version}',
        content
    )

    # 更新日期
    current_date = datetime.now().strftime('%Y-%m')
    new_content = re.sub(
        r'最后更新：\d{4}-\d{2}',
        f'最后更新：{current_date}',
        new_content
    )

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True


def update_version_files(root_dir, new_version):
    """更新所有包含版本信息的文件"""
    files_to_update = [
        root_dir / "pyproject.toml",
        root_dir / "README.md",
    ]

    updated = []
    for filepath in files_to_update:
        if filepath.exists():
            if "pyproject" in str(filepath):
                update_pyproject_version(filepath, new_version)
            else:
                update_readme_version(filepath, new_version)
            updated.append(filepath.name)
            print(f"  已更新：{filepath.name} -> v{new_version}")

    return updated


def main(increment_type='patch'):
    """主函数"""
    print("\n" + "=" * 60)
    print("  版本号自动更新")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent
    pyproject_path = root_dir / "pyproject.toml"
    readme_path = root_dir / "README.md"

    # 获取当前版本
    current_version = get_current_version(pyproject_path)
    print(f"\n  当前版本：v{current_version}")

    # 计算新版本
    new_version = increment_version(current_version, increment_type)
    print(f"  新版本：v{new_version}")

    # 更新文件
    print("\n  更新文件中...")
    updated_files = update_version_files(root_dir, new_version)

    print(f"\n  已更新 {len(updated_files)} 个文件：{', '.join(updated_files)}")
    print("\n" + "=" * 60)

    return new_version


if __name__ == "__main__":
    import sys
    increment_type = sys.argv[1] if len(sys.argv) > 1 else 'patch'
    main(increment_type)
