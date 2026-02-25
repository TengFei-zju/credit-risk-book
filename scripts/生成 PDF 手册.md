# 生成 PDF 手册

## 快速开始

### 步骤 1: 安装 PDF 生成库

选择以下任一方式：

**方式 A: 使用 fpdf2（推荐）**
```bash
pip install fpdf2
```

**方式 B: 使用 reportlab**
```bash
pip install reportlab
```

### 步骤 2: 运行生成脚本

```bash
python scripts/generate_pdf.py
```

输出文件：`信贷风控建模手册.pdf`

---

## 网络问题解决方案

如果 `pip install` 失败（SSL/代理错误），使用离线安装：

### 方法 1: 使用镜像源
```bash
pip install fpdf2 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 方法 2: 离线下载安装
1. 在有网络的机器上下载：
   ```bash
   pip download fpdf2 -d ./pkgs
   ```

2. 将 `pkgs` 文件夹复制到目标机器

3. 在目标机器安装：
   ```bash
   pip install --no-index --find-links=./pkgs fpdf2
   ```

### 方法 3: 手动下载 wheel 文件
1. 访问 https://pypi.org/project/fpdf2/#files
2. 下载 `fpdf2-*.whl` 文件
3. 本地安装：`pip install fpdf2-2.8.3-py2.py3-none-any.whl`

---

## 故障排除

### 问题：提示"未找到 PDF 生成库"
**解决**: 未安装 fpdf2 或 reportlab，按上述步骤安装

### 问题：PDF 中中文显示为方框
**解决**: 脚本会自动查找系统中文字体，确保系统已安装中文字体
- Windows: 默认有宋体/黑体
- macOS: 默认有苹方
- Linux: 安装文泉驿字体 `sudo apt install fonts-wqy-zenhei`

### 问题：git commit 时 hook 报错
**解决**: 先手动安装 PDF 库，然后 commit 时 hook 会自动生成

---

## 输出说明

生成的 PDF 包含：
- 封面页（标题、作者、版本）
- 所有章节内容（纯文本格式，代码块简化显示）

文件格式：A4，自动分页，带页眉页脚
