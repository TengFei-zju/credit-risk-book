#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图书 PDF 生成器
将所有 Markdown 章节合并并转换为 PDF 文件

使用方法：
    python scripts/generate_pdf.py [--html-only]

参数：
    --html-only  仅生成 HTML 文件，不生成 PDF
"""

import os
import re
import sys
import webbrowser
from pathlib import Path
from datetime import datetime

# 尝试导入 markdown 相关库
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# 尝试导入 PDF 生成库
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# 尝试导入中文支持
try:
    from fpdf import UnicodeMixin
    HAS_UNICODE = True
except ImportError:
    HAS_UNICODE = False


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


def convert_md_to_html(content):
    """将 Markdown 转换为 HTML"""
    if not MARKDOWN_AVAILABLE:
        # 简单转义
        return content.replace('<', '&lt;').replace('>', '&gt;')

    html = markdown.markdown(
        content,
        extensions=[
            'tables',
            'fenced_code',
            'nl2br',
            'codehilite',
        ],
        output_format='html5'
    )
    return html


def process_markdown_content(content, title=""):
    """处理 Markdown 内容，转换为适合 PDF 的格式"""
    lines = content.split('\n')
    processed = []
    in_code_block = False
    code_lines = []

    for line in lines:
        # 处理代码块
        if line.startswith('```'):
            if in_code_block:
                processed.append('[/code]')
                in_code_block = False
            else:
                processed.append('[code]')
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append('    ' + line)
            continue

        # 处理标题
        if line.startswith('######'):
            processed.append(f'\n###### {line[6:].strip()}\n')
        elif line.startswith('#####'):
            processed.append(f'\n##### {line[5:].strip()}\n')
        elif line.startswith('####'):
            processed.append(f'\n#### {line[4:].strip()}\n')
        elif line.startswith('###'):
            processed.append(f'\n### {line[3:].strip()}\n')
        elif line.startswith('##'):
            processed.append(f'\n## {line[2:].strip()}\n')
        elif line.startswith('#'):
            processed.append(f'\n# {line[1:].strip()}\n')
        else:
            processed.append(line)

    if code_lines:
        processed.append('\n'.join(code_lines))
        processed.append('[/code]\n')

    return '\n'.join(processed)


def create_html_document(content, title="信贷风控建模：打工人手册"):
    """创建完整的 HTML 文档"""
    version = datetime.now().strftime('%Y-%m')

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        @media print {{
            @page {{
                size: A4;
                margin: 2.5cm;
                @top-center {{
                    content: "{title}";
                    font-size: 9pt;
                    color: #666;
                }}
                @bottom-center {{
                    content: "第 " counter(page) " 页";
                    font-size: 9pt;
                    color: #666;
                }}
            }}
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            font-family: "Microsoft YaHei", "SimSun", "Source Han Sans CN", sans-serif;
            line-height: 1.8;
            color: #333;
            font-size: 11pt;
            max-width: 210mm;
            margin: 0 auto;
            padding: 20px;
        }}

        .cover {{
            text-align: center;
            padding: 4cm 2cm;
            page-break-after: always;
            background: linear-gradient(135deg, #0a1628 0%, #152642 100%);
            color: white;
            margin: -20px -20px 20px -20px;
            border-radius: 8px;
        }}

        .cover h1 {{
            font-size: 42pt;
            font-weight: bold;
            color: #ffffff;
            margin: 2cm 0 1cm 0;
            border: none;
        }}

        .cover .subtitle {{
            font-size: 20pt;
            color: #ffb84c;
            margin-bottom: 2cm;
        }}

        .cover .meta {{
            font-size: 12pt;
            color: #c0c0c0;
            margin-top: 3cm;
        }}

        .cover .tags {{
            margin-top: 2cm;
        }}

        .cover .tag {{
            display: inline-block;
            background: rgba(0, 212, 255, 0.2);
            border: 1px solid #00d4ff;
            padding: 8px 16px;
            margin: 5px;
            border-radius: 20px;
            font-size: 11pt;
        }}

        h1 {{
            color: #1a365d;
            border-bottom: 3px solid #2c5282;
            padding-bottom: 0.5em;
            margin-top: 1.5em;
            page-break-after: avoid;
            font-size: 24pt;
        }}

        h2 {{
            color: #2c5282;
            margin-top: 1.5em;
            page-break-after: avoid;
            font-size: 18pt;
        }}

        h3 {{
            color: #2c5282;
            margin-top: 1em;
            font-size: 14pt;
        }}

        a {{
            color: #3182ce;
            text-decoration: none;
        }}

        code {{
            background-color: #f7fafc;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Consolas", "Monaco", "Courier New", monospace;
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

        .chapter-start {{
            page-break-before: always;
            margin-top: 2em;
            padding-top: 2em;
        }}

        .chapter-start:first-child {{
            page-break-before: auto;
        }}

        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1em auto;
        }}

        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 2em 0;
        }}

        .print-button {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: #3182ce;
            color: white;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}

        .print-button:hover {{
            background: #2c5282;
        }}

        @media print {{
            .print-button {{
                display: none;
            }}
            body {{
                max-width: none;
                padding: 0;
            }}
            .cover {{
                border-radius: 0;
                margin: 0;
            }}
        }}
    </style>
</head>
<body>
    <button class="print-button" onclick="window.print()">📄 打印为 PDF</button>

    <div class="cover">
        <h1>信贷风控建模</h1>
        <div class="subtitle">打工人手册</div>
        <div style="color: #90a4ae; font-size: 14pt;">Credit Risk Modeling: A Practical Guide</div>

        <div class="tags">
            <span class="tag">特征工程</span>
            <span class="tag">机器学习</span>
            <span class="tag">图神经网络</span>
            <span class="tag">序列模型</span>
            <span class="tag">Kaggle 金牌方案</span>
        </div>

        <div class="meta">
            <div>作者：汪叽意且</div>
            <div>版本：v0.2 · {version}</div>
            <div style="margin-top: 1em; font-size: 11pt;">从数据清洗到模型部署的完整实战指南</div>
        </div>
    </div>

    {content}

    <script>
        // 自动转换代码块
        document.querySelectorAll('pre').forEach(pre => {{
            if (!pre.querySelector('code')) {{
                const code = document.createElement('code');
                code.innerHTML = pre.innerHTML;
                pre.innerHTML = '';
                pre.appendChild(code);
            }}
        }});
    </script>
</body>
</html>
"""


def merge_chapters():
    """合并所有章节"""
    print("=" * 60)
    print("合并图书章节...")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent

    merged_content = []
    chapter_order = get_chapter_order()

    for chapter in chapter_order:
        filepath = root_dir / chapter
        print(f"  处理：{chapter}")
        content = read_markdown_file(filepath)
        if content:
            # 处理内容
            content = process_markdown_content(content)

            # 为章节添加分隔
            if chapter.startswith("chapters/"):
                merged_content.append(f"\n\n<div class='chapter-start'></div>\n")
            merged_content.append(convert_md_to_html(content))

    return '\n'.join(merged_content)


def generate_pdf(html_only=False):
    """生成 PDF 文件"""
    print("\n" + "=" * 60)
    print("生成 PDF 文件...")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent

    # 合并章节并转换为 HTML
    print("\n  合并章节并转换为 HTML...")
    html_content = merge_chapters()

    # 创建完整的 HTML 文档
    full_html = create_html_document(html_content)

    # 保存 HTML 文件
    html_output = root_dir / "信贷风控建模手册.html"
    with open(html_output, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"  HTML 文件：{html_output}")

    if html_only:
        print("\n  已生成 HTML 文件（--html-only 模式）")
        return True

    # 尝试生成 PDF
    pdf_output = root_dir / "信贷风控建模手册.pdf"

    if FPDF_AVAILABLE:
        print("\n  使用 fpdf2 生成 PDF...")
        try:
            # 使用 fpdf2 生成 PDF
            pdf = FPDF()
            pdf.add_page()

            # 添加中文字体支持（需要系统中安装中文字体）
            font_path = "C:/Windows/Fonts/simsun.ttc"
            if os.path.exists(font_path):
                pdf.add_font('SimSun', '', font_path, uni=True)
                pdf.set_font('SimSun', '', 12)
            else:
                # 尝试其他中文字体
                font_candidates = [
                    "C:/Windows/Fonts/msyh.ttc",
                    "C:/Windows/Fonts/simhei.ttf",
                    "/System/Library/Fonts/PingFang.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                ]
                font_added = False
                for fp in font_candidates:
                    if os.path.exists(fp):
                        pdf.add_font('Chinese', '', fp, uni=True)
                        pdf.set_font('Chinese', '', 12)
                        font_added = True
                        break
                if not font_added:
                    pdf.set_font('Arial', '', 12)
                    print("  警告：未找到中文字体，PDF 可能无法正确显示中文")

            # 添加内容（简化版）
            pdf.multi_cell(0, 10, "信贷风控建模：打工人手册\n\nHTML 文件已生成，请使用浏览器打开并打印为 PDF 以获得最佳效果。")

            pdf.output(str(pdf_output))
            print(f"  PDF 文件：{pdf_output}")
        except Exception as e:
            print(f"  错误：PDF 生成失败 - {e}")
    else:
        print("\n  提示：未安装 fpdf2 库")
        print("  安装方法：pip install fpdf2")

    print("\n" + "=" * 60)
    print("  推荐使用以下方式生成 PDF：")
    print("  1. 用浏览器打开 HTML 文件")
    print("  2. 按 Ctrl+P (或点击页面上的打印按钮)")
    print("  3. 选择'另存为 PDF'")
    print("=" * 60)

    # 自动用浏览器打开
    try:
        print("\n  正在用浏览器打开 HTML 文件...")
        webbrowser.open(f'file:///{html_output.absolute()}')
    except:
        pass

    return True


def main():
    """主函数"""
    html_only = '--html-only' in sys.argv

    print("\n" + "=" * 60)
    print("  信贷风控建模：打工人手册 - PDF 生成器")
    print("=" * 60)

    generate_pdf(html_only)

    print("\n完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
