"""
信贷风控建模：打工人手册 - 封面生成器
设计风格：数据之盾 (The Data Shield)
纯 Pillow 实现，不依赖 matplotlib

优化版本：
- 简化背景网格
- 放大加粗标题
- 放大副标题
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# 配置
WIDTH, HEIGHT = 800, 1200
OUTPUT_DIR = Path(__file__).parent

# 配色方案
COLORS = {
    'deep_blue': (10, 22, 40),
    'mid_blue': (21, 38, 66),
    'light_blue': (27, 58, 95),
    'amber_gold': (255, 184, 76),
    'cyan_data': (0, 212, 255),
    'white': (255, 255, 255),
    'gray_text': (136, 153, 166)
}

def create_shield_path(cx, cy, width, height, num_points=100):
    """创建盾形路径点"""
    points = []

    # 顶部圆弧（从左到右）
    for i in range(num_points // 3):
        t = (i / (num_points // 3 - 1)) * np.pi
        x = cx - width/2 + width * (0.5 + 0.5 * np.cos(np.pi - t))
        y = cy - height/2 + 25 * np.sin(t)
        points.append((x, y))

    # 右侧斜线
    for i in range(num_points // 3):
        t = i / (num_points // 3 - 1)
        x = cx + width/2 - t * 15
        y = cy - height/2 + 25 + t * (height * 0.4)
        points.append((x, y))

    # 底部尖角
    points.append((cx, cy + height/2))

    # 左侧斜线
    for i in range(num_points // 3):
        t = i / (num_points // 3 - 1)
        x = cx - width/2 + t * 15
        y = cy - height/2 + 25 + (1-t) * (height * 0.4)
        points.append((x, y))

    return points

def draw_gradient_background(draw, width, height):
    """绘制渐变背景"""
    for y in range(height):
        t = y / height
        r = int(10 + t * 25)
        g = int(22 + t * 30)
        b = int(40 + t * 40)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

def draw_data_particles(draw, cx, cy, shield_w, shield_h, num_particles=80):
    """绘制数据流粒子 - 减少数量"""
    np.random.seed(42)

    for _ in range(num_particles):
        # 起始位置：封面边缘
        side = np.random.choice(['top', 'left', 'right', 'bottom'])
        if side == 'top':
            x = np.random.uniform(80, WIDTH-80)
            y = 50
        elif side == 'left':
            x = 50
            y = np.random.uniform(150, HEIGHT-200)
        elif side == 'right':
            x = WIDTH - 50
            y = np.random.uniform(150, HEIGHT-200)
        else:
            x = np.random.uniform(80, WIDTH-80)
            y = HEIGHT - 50

        # 目标位置：盾牌内部
        target_x = cx + np.random.uniform(-shield_w/3, shield_w/3)
        target_y = cy + np.random.uniform(-shield_h/3, shield_h/3)

        # 绘制粒子轨迹
        steps = 8
        for s in range(steps):
            interp_x = x + (target_x - x) * (s / steps)
            interp_y = y + (target_y - y) * (s / steps)
            size = np.random.uniform(2, 5) * (1 - s / steps * 0.5)

            # 颜色
            rand = np.random.random()
            if rand < 0.12:
                color = COLORS['amber_gold']  # 风险节点
            elif rand < 0.35:
                color = COLORS['cyan_data']  # 数据流
            else:
                color = COLORS['white']

            alpha = int(np.random.uniform(60, 180))
            color_with_alpha = (color[0], color[1], color[2], alpha)

            draw.ellipse([
                interp_x - size, interp_y - size,
                interp_x + size, interp_y + size
            ], fill=color_with_alpha)

def draw_network(draw, cx, cy, shield_w, shield_h):
    """绘制网络节点和连接"""
    # 节点位置
    nodes = [
        (cx, cy - shield_h * 0.3, COLORS['cyan_data']),
        (cx - shield_w * 0.25, cy - shield_h * 0.1, COLORS['amber_gold']),
        (cx + shield_w * 0.25, cy - shield_h * 0.1, COLORS['white']),
        (cx - shield_w * 0.3, cy + shield_h * 0.15, COLORS['white']),
        (cx, cy + shield_h * 0.2, COLORS['cyan_data']),
        (cx + shield_w * 0.3, cy + shield_h * 0.15, COLORS['white']),
    ]

    # 绘制连接线
    for i, (x1, y1, c1) in enumerate(nodes):
        for j, (x2, y2, c2) in enumerate(nodes):
            if j > i:
                if np.random.random() < 0.5:
                    alpha = int(np.random.uniform(30, 80))
                    line_color = (COLORS['cyan_data'][0], COLORS['cyan_data'][1], COLORS['cyan_data'][2], alpha)
                    draw.line([(x1, y1), (x2, y2)], fill=line_color, width=2)

    # 绘制节点
    for x, y, color in nodes:
        # 外圈光晕
        draw.ellipse([x-8, y-8, x+8, y+8], fill=(255, 255, 255, 40))
        # 核心节点
        size = 7 if color == COLORS['amber_gold'] else 5
        draw.ellipse([x-size, y-size, x+size, y+size], fill=color)

def draw_decorative_grid(draw, width, height):
    """绘制装饰性网格 - 简化版本"""
    # 只保留四角装饰线，移除全网格
    corner_len = 60

    # 左上角
    draw.line([(40, 60), (40 + corner_len, 60)], fill=COLORS['amber_gold'], width=3)
    draw.line([(40, 60), (40, 60 + corner_len)], fill=COLORS['amber_gold'], width=3)

    # 右下角
    draw.line([(width-40, height-60), (width-40-corner_len, height-60)], fill=COLORS['amber_gold'], width=3)
    draw.line([(width-40, height-60), (width-40, height-60-corner_len)], fill=COLORS['amber_gold'], width=3)

    # 非常淡的背景网格（可选，更低调）
    for x in [200, 400, 600]:
        draw.line([(x, 0), (x, height)], fill=(0, 212, 255, 8), width=1)

def draw_text_chinese(draw, text, size, color, position, font_name="msyh.ttc", bold=False):
    """绘制中文文字"""
    try:
        font = ImageFont.truetype(font_name, size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simsun.ttc", size)
        except:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    x = position[0] - text_w / 2
    draw.text((x, position[1]), text, fill=color, font=font)

def create_cover():
    """创建封面"""
    print("创建封面：数据之盾（优化版）...")

    # 创建 RGBA 图像
    img = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. 绘制渐变背景
    draw_gradient_background(draw, WIDTH, HEIGHT)

    # 2. 绘制简化装饰网格
    draw_decorative_grid(draw, WIDTH, HEIGHT)

    # 3. 盾牌参数
    shield_cx = WIDTH // 2
    shield_cy = HEIGHT // 2 - 50
    shield_w = 380
    shield_h = 420

    # 4. 绘制盾牌（半透明）
    shield_points = create_shield_path(shield_cx, shield_cy, shield_w, shield_h)

    # 盾牌填充
    shield_bg = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    shield_bg_draw = ImageDraw.Draw(shield_bg)
    shield_bg_draw.polygon(shield_points, fill=(27, 58, 95, 70))
    img = Image.alpha_composite(img, shield_bg)
    draw = ImageDraw.Draw(img)

    # 盾牌轮廓
    shield_outline = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
    outline_draw = ImageDraw.Draw(shield_outline)
    outline_draw.polygon(shield_points, outline=(0, 212, 255, 120), width=3)
    img = Image.alpha_composite(img, shield_outline)
    draw = ImageDraw.Draw(img)

    # 5. 绘制数据流粒子（减少数量）
    draw_data_particles(draw, shield_cx, shield_cy, shield_w, shield_h)

    # 6. 绘制网络节点
    draw_network(draw, shield_cx, shield_cy, shield_w, shield_h)

    # 7. 绘制标题 - 加粗放大
    draw_text_chinese(draw, "信贷风控建模", 72, COLORS['white'], (WIDTH/2, 180), bold=True)
    draw_text_chinese(draw, "打工人手册", 42, COLORS['amber_gold'], (WIDTH/2, 260))
    draw_text_chinese(draw, "Credit Risk Modeling: A Practical Guide", 14, COLORS['gray_text'], (WIDTH/2, 310))

    # 8. 特征标签
    tags = [
        ("特征工程", 200, 750),
        ("机器学习", 400, 750),
        ("图神经网络", 600, 750),
        ("序列模型", 300, 810),
        ("Kaggle 金牌方案", 520, 810),
    ]

    for tag, tx, ty in tags:
        try:
            tag_font = ImageFont.truetype("msyh.ttc", 18)  # 稍大一点
        except:
            tag_font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), tag, font=tag_font)
        tag_w = bbox[2] - bbox[0] + 36
        tag_h = 40

        # 圆角矩形背景 - 更明显的背景
        rect = [tx - tag_w/2, ty - tag_h/2, tx + tag_w/2, ty + tag_h/2]
        draw.rounded_rectangle(rect, radius=10, fill=COLORS['mid_blue'], outline=COLORS['amber_gold'], width=2)

        # 文字
        draw.text((tx - (bbox[2]-bbox[0])/2, ty - (bbox[3]-bbox[1])/2 - 2), tag, fill=COLORS['white'], font=tag_font)

    # 9. 分割线
    draw.line([(200, 880), (600, 880)], fill=COLORS['amber_gold'], width=2)

    # 10. 作者信息
    draw_text_chinese(draw, "作者", 14, COLORS['gray_text'], (WIDTH/2, 920))
    draw_text_chinese(draw, "汪叽意且", 22, COLORS['white'], (WIDTH/2, 950))
    draw.text((WIDTH/2 - 70, 990), "Version 0.2 · 2026", fill=(102, 102, 102), font=ImageFont.load_default())

    # 11. 底部标语
    draw_text_chinese(draw, "从数据清洗到模型部署的完整实战指南", 16, COLORS['gray_text'], (WIDTH/2, 1080))

    # 保存
    output_png = OUTPUT_DIR / "cover_new.png"

    # 转换为 RGB 并保存
    img_rgb = img.convert('RGB')
    img_rgb.save(output_png, "PNG", quality=95, dpi=(300, 300))
    print(f"封面已保存至：{output_png}")

    return output_png

if __name__ == "__main__":
    create_cover()
