#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图书 PDF 生成器 (使用 ReportLab)
高质量 PDF 生成，支持中文、代码块、表格

使用方法：
    python scripts/generate_pdf.py

输出：
    信贷风控建模手册.pdf
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime

# 导入 markdown
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("提示：markdown 库未安装，将使用原始文本")

# 导入 ReportLab
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.fonts import addMapping
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("错误：需要安装 ReportLab: pip install reportlab")
    sys.exit(1)


# 配置
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 2.5 * cm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


def find_chinese_fonts():
    """查找系统中的中文字体"""
    fonts = []
    font_candidates = [
        ("Microsoft YaHei", "C:/Windows/Fonts/msyh.ttc"),
        ("Microsoft YaHei Light", "C:/Windows/Fonts/msyhl.ttc"),
        ("SimSun", "C:/Windows/Fonts/simsun.ttc"),
        ("SimHei", "C:/Windows/Fonts/simhei.ttf"),
        ("SimKai", "C:/Windows/Fonts/simkai.ttf"),
        ("SimFan", "C:/Windows/Fonts/simfan.ttf"),
    ]

    for font_name, font_path in font_candidates:
        if os.path.exists(font_path):
            fonts.append((font_name, font_path))

    return fonts


def register_chinese_fonts():
    """注册中文字体到 ReportLab"""
    fonts = find_chinese_fonts()

    if not fonts:
        print("警告：未找到中文字体，PDF 可能无法正确显示中文")
        return False

    # 注册字体
    font_mapping = {
        'Chinese': None,  # 正文
        'Chinese-Bold': None,  # 粗体
        'Chinese-Title': None,  # 标题
    }

    for font_name, font_path in fonts:
        try:
            # 注册字体
            pdfmetrics.registerFont(TTFont(font_name, font_path))

            # 设置默认中文字体
            if font_mapping['Chinese'] is None:
                font_mapping['Chinese'] = font_name
            if 'SimHei' in font_name and font_mapping['Chinese-Bold'] is None:
                font_mapping['Chinese-Bold'] = font_name
            if 'SimKai' in font_name and font_mapping['Chinese-Title'] is None:
                font_mapping['Chinese-Title'] = font_name
        except Exception as e:
            print(f"警告：字体注册失败 {font_name}: {e}")

    # 如果没有找到粗体和标题字体，使用正文字体
    if font_mapping['Chinese-Bold'] is None:
        font_mapping['Chinese-Bold'] = font_mapping['Chinese']
    if font_mapping['Chinese-Title'] is None:
        font_mapping['Chinese-Title'] = font_mapping['Chinese-Bold']

    # 注册字体映射
    addMapping(font_mapping['Chinese'], 0, 0, font_mapping['Chinese'])
    addMapping(font_mapping['Chinese-Bold'], 1, 0, font_mapping['Chinese-Bold'])

    print(f"已注册中文字体：{font_mapping['Chinese']}")

    return font_mapping


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


def process_markdown_for_pdf(content):
    """将 Markdown 转换为适合 PDF 的格式"""
    if not MARKDOWN_AVAILABLE:
        return content

    # 使用 markdown 库转换
    html = markdown.markdown(
        content,
        extensions=['tables', 'fenced_code', 'nl2br'],
        output_format='html'
    )

    return html


def escape_for_pdf(text):
    """转义 PDF 特殊字符"""
    if not text:
        return ""
    # 转义 XML 特殊字符
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text


class BookPDF:
    """图书 PDF 生成器"""

    def __init__(self, output_path, font_mapping):
        self.output_path = output_path
        self.font_mapping = font_mapping
        self.doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN,
            title="信贷风控建模：打工人手册"
        )
        self.story = []
        self.styles = self._create_styles()

    def _create_styles(self):
        """创建样式"""
        styles = getSampleStyleSheet()

        # 封面标题
        styles.add(ParagraphStyle(
            name='CoverTitleCustom',
            parent=styles['Title'],
            fontName=self.font_mapping['Chinese-Title'],
            fontSize=36,
            textColor=colors.HexColor('#1a365d'),
            alignment=TA_CENTER,
            spaceAfter=30,
            leading=48,
        ))

        # 封面副标题
        styles.add(ParagraphStyle(
            name='CoverSubtitleCustom',
            parent=styles['Heading2'],
            fontName=self.font_mapping['Chinese-Bold'],
            fontSize=24,
            textColor=colors.HexColor('#ffb84c'),
            alignment=TA_CENTER,
            spaceAfter=20,
            leading=32,
        ))

        # 封面英文标题
        styles.add(ParagraphStyle(
            name='CoverEnglishCustom',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=14,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER,
            spaceAfter=40,
            leading=20,
        ))

        # 封面元信息
        styles.add(ParagraphStyle(
            name='CoverMetaCustom',
            parent=styles['Normal'],
            fontName=self.font_mapping['Chinese'],
            fontSize=12,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER,
            leading=20,
        ))

        # 一级标题
        styles.add(ParagraphStyle(
            name='Heading1Custom',
            parent=styles['Heading1'],
            fontName=self.font_mapping['Chinese-Bold'],
            fontSize=24,
            textColor=colors.HexColor('#1a365d'),
            spaceBefore=30,
            spaceAfter=20,
            leading=32,
        ))

        # 二级标题
        styles.add(ParagraphStyle(
            name='Heading2Custom',
            parent=styles['Heading2'],
            fontName=self.font_mapping['Chinese-Bold'],
            fontSize=18,
            textColor=colors.HexColor('#2c5282'),
            spaceBefore=24,
            spaceAfter=12,
            leading=26,
        ))

        # 三级标题
        styles.add(ParagraphStyle(
            name='Heading3Custom',
            parent=styles['Heading3'],
            fontName=self.font_mapping['Chinese-Bold'],
            fontSize=14,
            textColor=colors.HexColor('#2c5282'),
            spaceBefore=18,
            spaceAfter=8,
            leading=20,
        ))

        # 正文
        styles.add(ParagraphStyle(
            name='BodyTextCustom',
            parent=styles['Normal'],
            fontName=self.font_mapping['Chinese'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            alignment=TA_JUSTIFY,
            leading=18,
            firstLineIndent=0,
        ))

        # 代码块
        styles.add(ParagraphStyle(
            name='CodeBlockCustom',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            textColor=colors.HexColor('#2d3748'),
            backColor=colors.HexColor('#f7fafc'),
            borderWidth=1,
            borderColor=colors.HexColor('#e2e8f0'),
            leftIndent=10,
            rightIndent=10,
            leading=14,
        ))

        # 引用块
        styles.add(ParagraphStyle(
            name='BlockQuoteCustom',
            parent=styles['Normal'],
            fontName=self.font_mapping['Chinese'],
            fontSize=11,
            textColor=colors.HexColor('#2c5282'),
            backColor=colors.HexColor('#ebf8ff'),
            leftIndent=20,
            rightIndent=10,
            borderLeftColor=colors.HexColor('#4299e1'),
            borderLeftWidth=4,
            leading=18,
        ))

        return styles

    def add_cover(self):
        """添加封面"""
        # 标题
        self.story.append(Paragraph("信贷风控建模", self.styles['CoverTitleCustom']))
        self.story.append(Paragraph("打工人手册", self.styles['CoverSubtitleCustom']))
        self.story.append(Paragraph("Credit Risk Modeling: A Practical Guide", self.styles['CoverEnglishCustom']))

        # 标签
        tags = ["特征工程", "机器学习", "图神经网络", "序列模型", "Kaggle 金牌方案"]
        tags_table = Table([[tag for tag in tags]], colWidths=[4*cm]*len(tags))
        tags_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0a1628')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), self.font_mapping['Chinese']),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
        ]))
        self.story.append(Spacer(1, 2*cm))
        self.story.append(tags_table)

        # 作者信息
        self.story.append(Spacer(1, 4*cm))
        self.story.append(Paragraph("作者：汪叽意且", self.styles['CoverMetaCustom']))
        self.story.append(Paragraph(f"Version 0.2 · {datetime.now().strftime('%Y-%m')}", self.styles['CoverMetaCustom']))

        self.story.append(Spacer(1, 1*cm))
        self.story.append(Paragraph("从数据清洗到模型部署的完整实战指南", self.styles['CoverMetaCustom']))

        self.story.append(PageBreak())

    def add_chapter_title(self, title):
        """添加章节标题"""
        self.story.append(Spacer(1, 2*cm))
        self.story.append(Paragraph(title, self.styles['Heading1Custom']))
        self.story.append(Spacer(1, 1*cm))

    def add_content(self, content):
        """添加内容"""
        # 将 HTML 内容分割成段落
        paragraphs = re.split(r'\n\n+', content)

        in_code_block = False
        code_content = []

        for para in paragraphs:
            if not para.strip():
                continue

            # 检测代码块
            if '<pre>' in para or '<code>' in para:
                # 提取代码内容
                code_match = re.search(r'<pre[^>]*>(.*?)</pre>|<code[^>]*>(.*?)</code>', para, re.DOTALL)
                if code_match:
                    code = code_match.group(1) or code_match.group(2)
                    # 移除 HTML 标签
                    code = re.sub(r'<[^>]+>', '', code)
                    code = code.strip()

                    # 添加代码块
                    code_para = Paragraph(escape_for_pdf(code), self.styles['CodeBlockCustom'])
                    self.story.append(code_para)
                    self.story.append(Spacer(1, 12))
                    continue

            # 检测标题
            if para.startswith('<h1>'):
                title = re.sub(r'<[^>]+>', '', para[4:-5])
                self.story.append(Paragraph(escape_for_pdf(title), self.styles['Heading1Custom']))
                continue
            elif para.startswith('<h2>'):
                title = re.sub(r'<[^>]+>', '', para[4:-5])
                self.story.append(Paragraph(escape_for_pdf(title), self.styles['Heading2Custom']))
                continue
            elif para.startswith('<h3>'):
                title = re.sub(r'<[^>]+>', '', para[4:-5])
                self.story.append(Paragraph(escape_for_pdf(title), self.styles['Heading3Custom']))
                continue

            # 检测引用
            if para.startswith('<blockquote>'):
                quote = re.sub(r'<[^>]+>', '', para[13:-14])
                quote_para = Paragraph(escape_for_pdf(quote), self.styles['BlockQuoteCustom'])
                self.story.append(quote_para)
                self.story.append(Spacer(1, 12))
                continue

            # 检测表格
            if '<table>' in para:
                self._add_table(para)
                continue

            # 检测列表
            if para.startswith('<ul>') or para.startswith('<ol>'):
                self._add_list(para)
                continue

            # 普通段落
            text = re.sub(r'<[^>]+>', '', para)
            text = text.strip()
            if text:
                para_obj = Paragraph(escape_for_pdf(text), self.styles['BodyTextCustom'])
                self.story.append(para_obj)
                self.story.append(Spacer(1, 6))

    def _add_table(self, html):
        """添加表格"""
        # 简单的表格解析
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
        if not rows:
            return

        table_data = []
        for row in rows:
            cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', row, re.DOTALL)
            if cells:
                row_data = []
                for cell in cells:
                    cell_text = re.sub(r'<[^>]+>', '', cell).strip()
                    row_data.append(escape_for_pdf(cell_text))
                table_data.append(row_data)

        if table_data:
            # 计算列宽
            num_cols = max(len(row) for row in table_data)
            col_widths = [CONTENT_WIDTH / num_cols] * num_cols

            table = Table(table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#edf2f7')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-0, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), self.font_mapping['Chinese-Bold']),
                ('FONTNAME', (0, 1), (-1, -1), self.font_mapping['Chinese']),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
            ]))
            self.story.append(table)
            self.story.append(Spacer(1, 12))

    def _add_list(self, html):
        """添加列表"""
        items = re.findall(r'<li[^>]*>(.*?)</li>', html, re.DOTALL)
        for item in items:
            text = re.sub(r'<[^>]+>', '', item).strip()
            list_para = Paragraph(f"• {escape_for_pdf(text)}", self.styles['BodyTextCustom'])
            self.story.append(list_para)
            self.story.append(Spacer(1, 4))

    def build(self):
        """构建 PDF"""
        self.doc.build(self.story)
        print(f"PDF 已保存到：{self.output_path}")
        print(f"文件大小：{self.output_path.stat().st_size / 1024:.1f} KB")


def merge_chapters():
    """合并所有章节"""
    print("=" * 60)
    print("合并图书章节...")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent
    chapters_data = []
    chapter_order = get_chapter_order()

    for chapter in chapter_order:
        filepath = root_dir / chapter
        print(f"  处理：{chapter}")
        content = read_markdown_file(filepath)
        if content:
            chapters_data.append({
                'name': chapter,
                'content': content
            })

    return chapters_data


def generate_pdf():
    """生成 PDF 文件"""
    print("\n" + "=" * 60)
    print("生成 PDF 文件...")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent
    pdf_output = root_dir / "信贷风控建模手册.pdf"

    if not REPORTLAB_AVAILABLE:
        print("\n错误：需要安装 ReportLab")
        print("运行：pip install reportlab")
        return False

    # 注册中文字体
    font_mapping = register_chinese_fonts()
    if not font_mapping:
        return False

    # 合并章节
    chapters = merge_chapters()

    # 创建 PDF
    print("\n  正在生成 PDF...")
    pdf = BookPDF(pdf_output, font_mapping)

    # 添加封面
    pdf.add_cover()

    # 添加章节内容
    for chapter in chapters:
        if chapter['name'] == "README.md":
            # README 内容作为前言
            content = process_markdown_for_pdf(chapter['content'])
            # 跳过封面信息
            lines = content.split('\n')
            content = '\n'.join(lines[10:])  # 跳过前 10 行
            pdf.add_chapter_title("前言")
            pdf.add_content(content)
            pdf.story.append(PageBreak())
        else:
            # 提取章节标题
            content = process_markdown_for_pdf(chapter['content'])
            title_match = re.search(r'<h1[^>]*>(.*?)</h1>', content)
            if title_match:
                title = re.sub(r'<[^>]+>', '', title_match.group(0))
                pdf.add_chapter_title(title)
                pdf.add_content(content)
                pdf.story.append(PageBreak())

    # 构建 PDF
    pdf.build()

    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  信贷风控建模：打工人手册 - PDF 生成器 (ReportLab 版)")
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
