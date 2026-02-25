# 自动化构建脚本使用说明

## 功能概述

本项目包含一套自动化构建脚本，每次 git commit 时自动执行以下操作：

1. **自动更新版本号** - 递增 pyproject.toml 和 README.md 中的版本号
2. **生成 PDF 文件** - 将所有 Markdown 章节合并并转换为 PDF 格式
3. **自动添加到暂存区** - 生成的 PDF 会自动添加到 git 暂存区，下次 commit 时包含

## 文件结构

```
scripts/
├── build_book.py          # 主构建脚本（协调所有步骤）
├── update_version.py      # 版本号更新脚本
├── generate_pdf.py        # PDF/HTML 生成脚本（支持手动触发）
└── convert_drawio_to_svg.py

.git/hooks/
└── post-commit            # Git hook（自动触发）

requirements-pdf.txt       # PDF 生成依赖库
scripts/README_BUILD.md    # 本说明文档
```

## 安装依赖

```bash
pip install -r requirements-pdf.txt
```

必需库：
- `markdown` - Markdown 转 HTML（已自动安装）

可选库（如需直接生成 PDF）：
- `fpdf2` - 轻量级 PDF 生成库
- `xhtml2pdf` - HTML 转 PDF 库

## 手动运行

### 仅更新版本号

```bash
python scripts/update_version.py
```

可选参数：
- `patch`（默认）- 修订号 +1：0.1.0 → 0.1.1
- `minor` - 次版本号 +1：0.1.0 → 0.1.1
- `major` - 主版本号 +1：0.1.0 → 1.0.0

```bash
python scripts/update_version.py minor
```

### 仅生成 PDF

```bash
python scripts/generate_pdf.py
```

### 运行完整构建

```bash
python scripts/build_book.py
```

## Git Hook 配置

### 启用自动构建

Git hook 已配置在 `.git/hooks/post-commit`，无需额外配置。

每次执行 `git commit` 后，hook 会自动触发并运行构建脚本。

### 禁用自动构建

如果暂时不想自动构建，可以：

1. 重命名 hook 文件：
   ```bash
   mv .git/hooks/post-commit .git/hooks/post-commit.disabled
   ```

2. 或者删除执行权限：
   ```bash
   chmod -x .git/hooks/post-commit
   ```

## 输出文件

构建完成后会生成以下文件：

| 文件 | 说明 |
|------|------|
| `信贷风控建模手册.pdf` | 完整的书籍 PDF |
| `book_output.html` | 中间 HTML 文件（可用于浏览器打印） |

## 版本号规则

版本格式：`主版本。次版本.修订号`

- **修订号**（patch）- 小的修复和更新
- **次版本号**（minor）- 新功能添加
- **主版本号**（major）- 重大更新

每次 commit 默认递增修订号。

## 故障排除

### PDF 生成失败

如果 PDF 生成失败，请检查：

1. 是否安装了依赖库：`pip install -r requirements-pdf.txt`
2. 检查 Python 编码设置（Windows 可能需要设置 UTF-8）

### Hook 未触发

检查 hook 文件是否存在并有执行权限：

```bash
ls -la .git/hooks/post-commit
```

## 注意事项

1. **Windows 用户** - 如果使用 Git Bash，hook 正常工作；如果使用 PowerShell，可能需要调整 hook 脚本

2. **PDF 文件大小** - 包含所有章节的 PDF 可能较大，建议按需 commit

3. **构建时间** - PDF 生成需要数秒至数十秒，大量章节时请预留时间

## 自定义

### 修改章节顺序

编辑 `scripts/generate_pdf.py` 中的 `get_chapter_order()` 函数。

### 修改 PDF 样式

编辑 `scripts/generate_pdf.py` 中的 `create_pdf_html()` 函数内的 CSS 样式。
