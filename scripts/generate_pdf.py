#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图书 PDF 生成器
将所有 Markdown 章节合并并转换为 PDF 文件
"""

import os
import re
from pathlib import Path
from datetime import datetime

# 尝试导入 markdown 相关库
try:
    import markdown
    from markdown.extensions.tables import TableExtension
    from markdown.extensions.fenced_code import FencedCodeExtension
    from markdown.extensions.toc import TocExtension
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# 尝试导入 PDF 生成库
try:
    from xhtml2pdf import pisa
    XHTML2PDF_AVAILABLE = True
except ImportError:
    XHTML2PDF_AVAILABLE = False


def get_chapter_order():
    """定义章节顺序"""
    return [
        "README.md",
        "chapters/01_industry_overview.md",
        "chapters/02_business_understanding.md",
        "chapters/03_data_system.md",
        "chapters/04_feature_engineering.md",
        "chapters/05_scorecard.md",
        "chapters/06_machine_learning.md",
        "chapters/07_model_evaluation.md",
        "chapters/08_model_deployment.md",
        "chapters/09_model_monitoring.md",
        "chapters/10_strategy_decision.md",
        "chapters/11_anti_fraud.md",
        "chapters/12_collection.md",
        "chapters/13_graph_models.md",
        "chapters/14_sequence_models.md",
        "chapters/15_llm_in_risk.md",
        "chapters/17_references.md",
        "chapters/appendix.md",
    ]


def read_markdown_file(filepath):
    """读取 Markdown 文件内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"  警告：文件不存在 {filepath}")
        return ""
    except Exception as e:
        print(f"  错误：读取 {filepath} 失败 - {e}")
        return ""


def extract_front_matter(content):
    """提取 README 中的前言信息（作者、版本等）"""
    lines = content.split('\n')
    front_matter = []
    in_front_matter = False

    for line in lines:
        if line.startswith('# 信贷风控建模'):
            in_front_matter = True
        if in_front_matter:
            front_matter.append(line)
            if line.strip() == '' and len(front_matter) > 5:
                # 找到第一个空行且已经收集了足够多行
                if any('作者' in l or '版本' in l for l in front_matter):
                    break

    return '\n'.join(front_matter[:10])  # 限制最多 10 行


def convert_md_to_html(content, title=""):
    """将 Markdown 转换为 HTML"""
    if not MARKDOWN_AVAILABLE:
        # 如果没有 markdown 库，返回简单的 HTML
        return f"<h1>{title}</h1><pre>{content}</pre>"

    html = markdown.markdown(
        content,
        extensions=[
            'tables',
            'fenced_code',
            'toc',
            'nl2br',
        ],
        output_format='html5'
    )
    return html


def create_pdf_html(content, title="信贷风控建模：打工人手册"):
    """创建完整的 HTML 文档用于 PDF 生成"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        @page {{
            size: A4;
            margin: 2.5cm;
            @top-center {{
                content: "信贷风控建模：打工人手册";
                font-size: 9pt;
                color: #666;
            }}
            @bottom-center {{
                content: "页码：" counter(page);
                font-size: 9pt;
                color: #666;
            }}
        }}

        body {{
            font-family: "Microsoft YaHei", "SimSun", sans-serif;
            line-height: 1.8;
            color: #333;
            font-size: 11pt;
        }}

        h1 {{
            color: #1a365d;
            border-bottom: 2px solid #2c5282;
            padding-bottom: 0.5em;
            margin-top: 1em;
            page-break-after: avoid;
        }}

        h2, h3, h4, h5, h6 {{
            color: #2c5282;
            margin-top: 1.5em;
            page-break-after: avoid;
        }}

        h1 {{ font-size: 24pt; }}
        h2 {{ font-size: 18pt; }}
        h3 {{ font-size: 14pt; }}

        a {{
            color: #3182ce;
            text-decoration: none;
        }}

        code {{
            background-color: #f7fafc;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: "Consolas", "Monaco", monospace;
            font-size: 9pt;
        }}

        pre {{
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 1em;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.5;
        }}

        pre code {{
            background: none;
            padding: 0;
            color: inherit;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            font-size: 10pt;
        }}

        th, td {{
            border: 1px solid #cbd5e0;
            padding: 0.75em;
            text-align: left;
        }}

        th {{
            background-color: #edf2f7;
            font-weight: bold;
        }}

        tr:nth-child(even) {{
            background-color: #f7fafc;
        }}

        blockquote {{
            margin: 1em 0;
            padding: 0.5em 1em;
            border-left: 4px solid #4299e1;
            background-color: #ebf8ff;
            color: #2c5282;
        }}

        ul, ol {{
            margin: 0.5em 0;
            padding-left: 2em;
        }}

        li {{
            margin: 0.3em 0;
        }}

        .cover-page {{
            text-align: center;
            padding: 3cm 2cm;
            page-break-after: always;
        }}

        .cover-title {{
            font-size: 36pt;
            font-weight: bold;
            color: #1a365d;
            margin: 2cm 0 1cm 0;
        }}

        .cover-subtitle {{
            font-size: 18pt;
            color: #2c5282;
            margin-bottom: 3cm;
        }}

        .cover-meta {{
            font-size: 12pt;
            color: #718096;
        }}

        .chapter-start {{
            page-break-before: always;
        }}

        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1em auto;
        }}
    </style>
</head>
<body>
{content}
</body>
</html>
"""


def merge_chapters():
    """合并所有章节"""
    print("=" * 60)
    print("合并图书章节...")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent
    chapters_dir = root_dir / "chapters"

    merged_content = []
    chapter_order = get_chapter_order()

    for chapter in chapter_order:
        filepath = root_dir / chapter
        print(f"  处理：{chapter}")
        content = read_markdown_file(filepath)
        if content:
            # 为章节添加分隔
            if chapter.startswith("chapters/"):
                # 提取章节标题
                first_line = content.split('\n')[0] if content else ""
                if first_line.startswith('#'):
                    merged_content.append(f"\n\n<div class='chapter-start'></div>\n")
            merged_content.append(content)

    return '\n\n---\n\n'.join(merged_content)


def generate_pdf():
    """生成 PDF 文件"""
    print("\n" + "=" * 60)
    print("生成 PDF 文件...")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent

    # 合并章节
    merged_md = merge_chapters()

    # 提取前言信息
    front_matter = extract_front_matter(merged_md)

    # 转换为 HTML
    print("\n  转换为 HTML...")
    if MARKDOWN_AVAILABLE:
        html_content = convert_md_to_html(merged_md)
    else:
        # 简单处理：替换一些基本格式
        html_content = merged_md.replace('<', '&lt;').replace('>', '&gt;')
        html_content = f"<h1>信贷风控建模：打工人手册</h1>\n<pre>{html_content}</pre>"

    # 创建完整的 HTML 文档
    full_html = create_pdf_html(html_content)

    # 保存 HTML 中间文件
    html_output = root_dir / "book_output.html"
    with open(html_output, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"  HTML 临时文件：{html_output}")

    # 生成 PDF
    pdf_output = root_dir / "信贷风控建模手册.pdf"

    if XHTML2PDF_AVAILABLE:
        print("  使用 xhtml2pdf 生成 PDF...")
        try:
            with open(pdf_output, 'wb') as f:
                pisa.CreatePDF(full_html, dest=f, encoding='utf-8')
            print(f"  PDF 已生成：{pdf_output}")
            return True
        except Exception as e:
            print(f"  错误：PDF 生成失败 - {e}")
    else:
        print("  警告：xhtml2pdf 未安装，仅生成 HTML 文件")
        print("  安装方法：pip install xhtml2pdf")
        print(f"  您可以使用浏览器打开 {html_output} 并打印为 PDF")

    return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  信贷风控建模：打工人手册 - PDF 生成器")
    print("=" * 60)

    success = generate_pdf()

    if success:
        print("\n" + "=" * 60)
        print("  PDF 生成成功!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("  注意：需要安装依赖库以生成 PDF")
        print("  运行：pip install xhtml2pdf markdown")
        print("=" * 60)


if __name__ == "__main__":
    main()
