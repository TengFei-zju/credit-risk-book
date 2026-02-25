#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图书 PDF 生成器
将所有 Markdown 章节合并并转换为 PDF 文件

使用方法：
    python scripts/generate_pdf.py

依赖安装（首次使用）：
    方式 1: pip install fpdf2
    方式 2: pip install reportlab

    如果网络有问题，可以：
    1. 在有网的机器下载：pip download fpdf2 -d ./pkgs
    2. 复制到目标机器：pip install --no-index --find-links=./pkgs fpdf2

输出：
    信贷风控建模手册.pdf
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime

# 尝试导入 markdown
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("警告：markdown 库未安装，代码块可能无法正确处理")

# 尝试导入 PDF 库
FPDF_AVAILABLE = False
REPORTLAB_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
    print("使用 FPDF2 生成 PDF")
except ImportError:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        REPORTLAB_AVAILABLE = True
        print("使用 ReportLab 生成 PDF")
    except ImportError:
        pass


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


def strip_markdown(content):
    """将 Markdown 转换为纯文本（去除格式）"""
    # 移除代码块，替换为占位符
    code_blocks = []
    def save_code_block(match):
        code_blocks.append(match.group(1))
        return f"\n[代码块 {len(code_blocks)}]\n"

    content = re.sub(r'```(?:\w+)?\n(.*?)```', save_code_block, content, flags=re.DOTALL)

    # 移除行内代码
    content = re.sub(r'`([^`]+)`', r'\1', content)

    # 处理标题
    content = re.sub(r'^######\s*(.+)$', r'\1', content, flags=re.MULTILINE)
    content = re.sub(r'^#####\s*(.+)$', r'\1', content, flags=re.MULTILINE)
    content = re.sub(r'^####\s*(.+)$', r'\1', content, flags=re.MULTILINE)
    content = re.sub(r'^###\s*(.+)$', r'=== \1 ===', content, flags=re.MULTILINE)
    content = re.sub(r'^##\s*(.+)$', r'>> \1', content, flags=re.MULTILINE)
    content = re.sub(r'^#\s*(.+)$', r'>>> \1 <<<', content, flags=re.MULTILINE)

    # 移除图片
    content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[\1]', content)

    # 处理链接
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)

    # 处理粗体和斜体
    content = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', content)
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    content = re.sub(r'\*([^*]+)\*', r'\1', content)

    # 处理列表
    content = re.sub(r'^[\-\*+]\s+', '  • ', content, flags=re.MULTILINE)
    content = re.sub(r'^\d+\.\s+', '  ', content, flags=re.MULTILINE)

    # 处理引用
    content = re.sub(r'^>\s*', '  ', content, flags=re.MULTILINE)

    # 处理表格
    lines = content.split('\n')
    processed_lines = []
    in_table = False
    for line in lines:
        if re.match(r'^\|.*\|', line):
            # 移除表格分隔线
            if re.match(r'^\|[\s\-:|]+\|$', line):
                continue
            # 保留表格内容，移除多余竖线
            line = re.sub(r'\|', ' | ', line)
            line = line.strip()
            in_table = True
            processed_lines.append(line)
        else:
            if in_table:
                processed_lines.append('')  # 表格后加空行
                in_table = False
            processed_lines.append(line)

    content = '\n'.join(processed_lines)

    # 移除 HTML 标签
    content = re.sub(r'<[^>]+>', '', content)

    # 压缩多余空行
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()


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
            # 转换为纯文本
            text_content = strip_markdown(content)

            # 添加章节标题
            if chapter == "README.md":
                merged_content.append("信贷风控建模：打工人手册\n")
                merged_content.append("Credit Risk Modeling: A Practical Guide\n")
            elif chapter.startswith("chapters/"):
                # 提取章节标题
                first_line = text_content.split('\n')[0] if text_content else ""
                merged_content.append(f"\n\n{'='*40}\n")
                merged_content.append(f"{first_line}\n")
                merged_content.append(f"{'='*40}\n")
                text_content = '\n'.join(text_content.split('\n')[1:])

            merged_content.append(text_content)

    return '\n\n'.join(merged_content)


def find_chinese_font():
    """查找系统中的中文字体"""
    font_candidates = [
        # Windows
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simkai.ttf",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            return font_path
    return None


def generate_pdf_with_fpdf(text_content, output_path):
    """使用 FPDF2 生成 PDF"""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)

    # 添加封面
    pdf.add_page()

    # 查找中文字体
    font_path = find_chinese_font()

    if font_path:
        try:
            pdf.add_font('Chinese', '', font_path, uni=True)
            pdf.set_font('Chinese', '', 14)
            print(f"  使用中文字体：{font_path}")
        except Exception as e:
            print(f"  警告：字体加载失败 {e}")
            pdf.set_font('Arial', '', 12)
    else:
        print("  警告：未找到中文字体，使用英文字体")
        pdf.set_font('Arial', '', 12)

    # 封面内容
    pdf.set_font_size(28)
    pdf.cell(0, 20, 'Credit Risk Modeling', ln=True, align='C')
    pdf.set_font_size(20)
    pdf.cell(0, 15, 'A Practical Guide', ln=True, align='C')

    pdf.ln(30)
    pdf.set_font_size(12)
    pdf.cell(0, 10, 'Author: Wang Jiyi', ln=True, align='C')
    pdf.cell(0, 10, f'Version 0.2 - {datetime.now().strftime("%Y-%m")}', ln=True, align='C')

    # 内容页
    lines = text_content.split('\n')
    for line in lines:
        # 处理 Unicode 字符
        try:
            if font_path:
                pdf.set_font('Chinese', '', 11)
            else:
                # 检测是否包含中文
                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in line)
                if has_chinese:
                    # 跳过无法处理的行
                    continue

            # 处理长行自动换行
            pdf.multi_cell(0, 8, line.encode('latin-1', 'replace').decode('latin-1'))
        except Exception as e:
            # 跳过无法编码的行
            continue

    pdf.output(str(output_path))
    return True


def generate_pdf_with_reportlab(text_content, output_path):
    """使用 ReportLab 生成 PDF"""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.units import cm

    # 创建 PDF
    pdf_file = str(output_path)
    c = canvas.Canvas(pdf_file, pagesize=A4)
    width, height = A4

    # 查找并注册中文字体
    font_path = find_chinese_font()
    if font_path:
        try:
            pdfmetrics.registerFont(TTFont('Chinese', font_path))
            print(f"  使用中文字体：{font_path}")
            font_name = 'Chinese'
        except Exception as e:
            print(f"  警告：字体注册失败 {e}")
            font_name = 'Helvetica'
    else:
        print("  警告：未找到中文字体")
        font_name = 'Helvetica'

    # 封面
    c.setFont(font_name, 32)
    c.drawCentredString(width/2, height - 8*cm, 'Credit Risk Modeling')
    c.setFont(font_name, 24)
    c.drawCentredString(width/2, height - 10*cm, 'A Practical Guide')

    c.setFont(font_name, 14)
    c.drawCentredString(width/2, height - 14*cm, 'Author: Wang Jiyi')
    c.drawCentredString(width/2, height - 15.5*cm, f'Version 0.2 - {datetime.now().strftime("%Y-%m")}')

    c.showPage()

    # 内容页
    c.setFont(font_name, 11)
    y = height - 2.5*cm
    line_height = 14

    lines = text_content.split('\n')
    for line in lines:
        # 跳过空行
        if not line.strip():
            y -= line_height
            continue

        # 简单分页
        if y < 2*cm:
            c.showPage()
            c.setFont(font_name, 11)
            y = height - 2.5*cm

        # 处理长行
        max_chars = 45  # 每行最大字符数
        if len(line) > max_chars:
            # 自动换行
            words = line.split(' ')
            current_line = ''
            for word in words:
                if len(current_line + ' ' + word) <= max_chars:
                    current_line += (' ' if current_line else '') + word
                else:
                    if font_name == 'Chinese' or all(ord(c) < 128 for c in current_line):
                        c.drawString(2.5*cm, y, current_line[:80])  # 截断过长行
                    y -= line_height
                    current_line = word
            if current_line:
                if font_name == 'Chinese' or all(ord(c) < 128 for c in current_line):
                    c.drawString(2.5*cm, y, current_line[:80])
                y -= line_height
        else:
            # 只打印 ASCII 或中文
            try:
                c.drawString(2.5*cm, y, line[:80])
            except:
                pass
            y -= line_height

    c.save()
    return True


def generate_pdf():
    """生成 PDF 文件"""
    print("\n" + "=" * 60)
    print("生成 PDF 文件...")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent
    pdf_output = root_dir / "信贷风控建模手册.pdf"

    # 检查是否有 PDF 库
    if not FPDF_AVAILABLE and not REPORTLAB_AVAILABLE:
        print("\n错误：未找到 PDF 生成库")
        print("\n请安装以下任一库：")
        print("  方案 1: pip install fpdf2")
        print("  方案 2: pip install reportlab")
        print("\n或者手动安装到离线环境：")
        print("  1. 在有网络的机器下载：pip download fpdf2 -d ./pkgs")
        print("  2. 复制 pkgs 目录到目标机器")
        print("  3. pip install --no-index --find-links=./pkgs fpdf2")
        return False

    # 合并章节
    text_content = merge_chapters()

    # 生成 PDF
    print("\n  正在生成 PDF...")

    try:
        if FPDF_AVAILABLE:
            success = generate_pdf_with_fpdf(text_content, pdf_output)
        else:
            success = generate_pdf_with_reportlab(text_content, pdf_output)

        if success:
            print(f"  PDF 已生成：{pdf_output}")
            print(f"  文件大小：{pdf_output.stat().st_size / 1024:.1f} KB")
            return True
        else:
            print("  错误：PDF 生成失败")
            return False

    except Exception as e:
        print(f"  错误：{e}")
        import traceback
        traceback.print_exc()
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
        return 0
    else:
        print("\n" + "=" * 60)
        print("  PDF 生成失败")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
