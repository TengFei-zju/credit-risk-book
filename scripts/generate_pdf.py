#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图书 PDF 生成器 (使用 Pillow)
纯 Python 实现，无需额外依赖

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

# 尝试导入 markdown
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("提示：markdown 库未安装，将使用原始文本")

# 导入 Pillow
try:
    from PIL import Image, ImageDraw, ImageFont
    from PIL.ImageFont import FreeTypeFont, ImageFont as PilFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("错误：需要安装 Pillow: pip install pillow")
    sys.exit(1)


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
    """将 Markdown 转换为纯文本"""
    # 移除代码块
    content = re.sub(r'```(?:\w+)?\n.*?```', '[代码块]', content, flags=re.DOTALL)
    # 移除行内代码
    content = re.sub(r'`([^`]+)`', r'\1', content)
    # 处理标题
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)
    # 移除图片
    content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[\1]', content)
    # 处理链接
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
    # 处理粗体斜体
    content = re.sub(r'\*+([^*]+)\*+', r'\1', content)
    # 处理列表
    content = re.sub(r'^[\-\*+]\s+', '• ', content, flags=re.MULTILINE)
    content = re.sub(r'^\d+\.\s+', '  ', content, flags=re.MULTILINE)
    # 处理引用
    content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
    # 移除表格分隔线
    content = re.sub(r'^\|?[\s\-:|]+\|?\n', '', content, flags=re.MULTILINE)
    # 移除 HTML
    content = re.sub(r'<[^>]+>', '', content)
    # 压缩空行
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


def find_chinese_font(size=12):
    """查找系统中的中文字体"""
    font_candidates = [
        ("msyh.ttc", "Microsoft YaHei"),
        ("simsun.ttc", "SimSun"),
        ("simhei.ttf", "SimHei"),
        ("simkai.ttf", "SimKai"),
    ]

    for font_file, font_name in font_candidates:
        font_path = f"C:/Windows/Fonts/{font_file}"
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue

    # 尝试默认字体
    try:
        return ImageFont.load_default()
    except:
        return None


class SimplePDF:
    """简单的 PDF 生成器（使用 Pillow）"""

    def __init__(self, width=2480, height=3508, dpi=300):
        """A4 尺寸，300 DPI"""
        self.width = width
        self.height = height
        self.dpi = dpi
        self.pages = []
        self.margin_left = int(2.5 * dpi / 2.54)  # 2.5cm
        self.margin_right = int(2.5 * dpi / 2.54)
        self.margin_top = int(2.5 * dpi / 2.54)
        self.margin_bottom = int(2.5 * dpi / 2.54)

        # 字体
        self.font_title = find_chinese_font(24)
        self.font_heading = find_chinese_font(18)
        self.font_body = find_chinese_font(12)
        self.line_height = 18

        # 当前页面
        self.current_page = None
        self.current_draw = None
        self.current_y = self.margin_top

    def new_page(self):
        """创建新页面"""
        if self.current_page is not None:
            self.pages.append(self.current_page)

        self.current_page = Image.new('RGB', (self.width, self.height), 'white')
        self.current_draw = ImageDraw.Draw(self.current_page)
        self.current_y = self.margin_top

        # 添加页眉线
        self.current_draw.line([
            (self.margin_left, self.margin_top - 20),
            (self.width - self.margin_right, self.margin_top - 20)
        ], fill='#1a365d', width=3)

    def add_text(self, text, font=None, color=(0, 0, 0)):
        """添加文本"""
        if font is None:
            font = self.font_body

        lines = text.split('\n')
        for line in lines:
            if not line:
                self.current_y += self.line_height
                continue

            # 检查是否需要新页面
            if self.current_y + self.line_height > self.height - self.margin_bottom:
                self.new_page()

            # 处理长行自动换行
            max_width = self.width - self.margin_left - self.margin_right
            wrapped_lines = self._wrap_text(line, font, max_width)

            for wrapped_line in wrapped_lines:
                if self.current_y + self.line_height > self.height - self.margin_bottom:
                    self.new_page()

                self.current_draw.text(
                    (self.margin_left, self.current_y),
                    wrapped_line,
                    font=font,
                    fill=color
                )
                self.current_y += self.line_height

    def _wrap_text(self, text, font, max_width):
        """文本换行"""
        words = text.split(' ')
        lines = []
        current_line = ''

        for word in words:
            test_line = current_line + (' ' if current_line else '') + word
            bbox = self.current_draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [text]

    def add_heading(self, text, level=1):
        """添加标题"""
        if level == 1:
            font = self.font_title
            self.current_y += 20
        elif level == 2:
            font = self.font_heading
            self.current_y += 15
        else:
            font = self.font_body
            self.current_y += 10

        self.add_text(text, font=font, color=(26, 54, 93))
        self.current_y += 10

    def save(self, output_path):
        """保存为 PDF"""
        if self.current_page:
            self.pages.append(self.current_page)

        if not self.pages:
            print("没有内容可保存")
            return

        # 保存为 PDF
        self.pages[0].save(
            output_path,
            'PDF',
            resolution=self.dpi,
            save_all=True,
            append_images=self.pages[1:] if len(self.pages) > 1 else []
        )
        print(f"PDF 已保存到：{output_path}")


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
                merged_content.append(("COVER", ""))
            elif chapter.startswith("chapters/"):
                # 提取章节标题
                lines = text_content.split('\n')
                if lines and lines[0].strip():
                    merged_content.append(("CHAPTER", lines[0]))
                    text_content = '\n'.join(lines[1:])
                else:
                    merged_content.append(("SECTION", chapter))

            merged_content.append(("CONTENT", text_content))

    return merged_content


def generate_pdf():
    """生成 PDF 文件"""
    print("\n" + "=" * 60)
    print("生成 PDF 文件...")
    print("=" * 60)

    root_dir = Path(__file__).parent.parent
    pdf_output = root_dir / "信贷风控建模手册.pdf"

    if not PIL_AVAILABLE:
        print("\n错误：需要安装 Pillow")
        print("运行：pip install pillow")
        return False

    # 合并章节
    chapters = merge_chapters()

    # 创建 PDF
    print("\n  正在生成 PDF（这可能需要几分钟）...")
    pdf = SimplePDF()
    pdf.new_page()

    for item_type, content in chapters:
        if item_type == "COVER":
            # 封面
            pdf.current_draw = ImageDraw.Draw(pdf.current_page)

            # 标题
            title_font = find_chinese_font(48)
            subtitle_font = find_chinese_font(28)
            meta_font = find_chinese_font(16)

            y = pdf.height // 2 - 100

            # 主标题
            pdf.current_draw.text(
                (pdf.width // 2, y),
                "信贷风控建模",
                font=title_font,
                fill=(26, 54, 93)
            )
            y += 80

            # 副标题
            pdf.current_draw.text(
                (pdf.width // 2, y),
                "打工人手册",
                font=subtitle_font,
                fill=(255, 184, 76)
            )
            y += 60

            # 英文标题
            pdf.current_draw.text(
                (pdf.width // 2, y),
                "Credit Risk Modeling: A Practical Guide",
                font=meta_font,
                fill=(100, 100, 100)
            )
            y += 80

            # 作者信息
            pdf.current_draw.text(
                (pdf.width // 2, y),
                "作者：汪叽意且",
                font=meta_font,
                fill=(80, 80, 80)
            )
            y += 30
            pdf.current_draw.text(
                (pdf.width // 2, y),
                f"Version 0.2 · {datetime.now().strftime('%Y-%m')}",
                font=meta_font,
                fill=(80, 80, 80)
            )

            pdf.new_page()

        elif item_type == "CHAPTER":
            # 章节标题页
            pdf.add_text("\n\n\n")
            pdf.add_heading(content, level=1)
            pdf.add_text("\n\n")

        elif item_type == "CONTENT":
            # 内容
            pdf.add_text(content, font=pdf.font_body)

    # 保存
    pdf.save(str(pdf_output))
    print(f"  文件大小：{pdf_output.stat().st_size / 1024:.1f} KB")
    return True


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  信贷风控建模：打工人手册 - PDF 生成器 (Pillow 版)")
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
