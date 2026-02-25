# 第十五章 风控中的大模型应用

## 15.1 LLM在风控场景的定位

大语言模型（LLM）在信贷风控中**不是替代传统风控模型**，而是在特定场景发挥独特价值：

```
LLM适合的风控场景（擅长非结构化信息处理）：
├── 非结构化文本分析（财务报告、企业年报、新闻舆情）
├── 客户沟通辅助（智能催收、贷前问卷、客服）
├── 规则/策略文档的语义理解与维护
├── 风控报告自动生成
└── 内部知识问答（模型文档、策略手册）

LLM不适合的场景（需要精确数值推理）：
├── 核心信用评分（仍用传统ML）
├── 实时欺诈判断（延迟不满足要求）
└── 需要精确概率校准的场景
```

---

## 15.2 LLM用于非结构化数据分析

### 15.2.1 企业财务报告风险解读

```python
from anthropic import Anthropic

client = Anthropic()

def analyze_financial_report(report_text: str, company_name: str) -> dict:
    """
    使用LLM分析企业财务报告，提取风险信号
    适用于：小微企业贷款尽调、供应链金融
    """
    prompt = f"""
你是一位资深的企业信用分析师，请分析以下{company_name}的财务报告内容，
从信贷风险视角提取关键信息。

财务报告内容：
{report_text}

请按以下结构输出分析结果（JSON格式）：
{{
    "revenue_trend": "收入趋势（增长/平稳/下滑），并说明原因",
    "profitability": "盈利能力评估（健康/一般/堪忧）",
    "debt_risk": "负债风险（低/中/高），关键数据点",
    "cash_flow_health": "现金流状况",
    "red_flags": ["风险预警点列表"],
    "positive_signals": ["积极信号列表"],
    "overall_credit_risk": "综合信用风险评级（低/中低/中/中高/高）",
    "confidence": "分析置信度（0-1）",
    "analyst_note": "分析师备注（重要但报告中未明确体现的判断）"
}}
"""

    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    try:
        result = json.loads(message.content[0].text)
    except json.JSONDecodeError:
        result = {"raw_response": message.content[0].text}

    return result
```

### 15.2.2 舆情风险监控

```python
def analyze_news_risk(news_articles: list, entity_name: str) -> dict:
    """
    分析与借款企业/个人相关的新闻舆情风险
    """
    news_text = "\n\n".join([
        f"[{i+1}] 标题：{a['title']}\n内容：{a['content'][:500]}"
        for i, a in enumerate(news_articles)
    ])

    prompt = f"""
分析以下关于"{entity_name}"的新闻报道，评估其信贷风险相关信号。

新闻内容：
{news_text}

请输出：
1. 负面信号（法律纠纷、经营困难、高管变动等）
2. 正面信号（融资、扩张、获奖等）
3. 综合舆情风险级别：低/中/高
4. 需要人工核实的关键信息点
"""
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"entity": entity_name, "analysis": message.content[0].text}
```

---

## 15.3 LLM辅助催收

### 15.3.1 个性化催收话术生成

```python
def generate_collection_script(customer_profile: dict, overdue_info: dict,
                                previous_contacts: list) -> str:
    """
    根据客户画像生成个性化催收话术
    注意：仅作为辅助工具，人工催收员可自行调整
    """
    context = f"""
客户基本信息：
- 姓名：{customer_profile.get('name', '客户')}
- 逾期金额：{overdue_info['amount']}元
- 逾期天数：{overdue_info['dpd']}天
- 历史还款记录：{customer_profile.get('payment_history', '良好')}

历史联系记录：
{chr(10).join(previous_contacts[-3:]) if previous_contacts else '首次联系'}
"""

    prompt = f"""
你是一位专业、有温度的信贷催收专员。请根据以下客户信息，
生成一段专业、合规的电话催收开场白和话术框架。

要求：
1. 语气专业但不强硬，体现理解与帮助的态度
2. 符合监管要求（不得威胁、恐吓、骚扰）
3. 提供灵活的还款方案引导（分期、延期等）
4. 如客户有特殊困难，引导到客服处理通道

{context}

请生成：
1. 开场白（20字以内）
2. 说明来意话术
3. 灵活还款方案引导话术
4. 结束语
"""
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

---

## 15.4 LLM用于风控知识问答

### 15.4.1 RAG（检索增强生成）风控知识库

```python
# 构建风控内部知识问答系统
# 数据源：策略文档、模型说明书、监管规定

from anthropic import Anthropic
import json

client = Anthropic()

def build_risk_qa_system(knowledge_docs: list):
    """
    构建风控知识问答系统（简化版RAG）
    适用于：
    - 新员工培训问答
    - 模型文档查询
    - 策略一致性核查
    """
    # 实际场景中需要向量数据库（Faiss/Pinecone等）
    # 这里用简化的关键词检索示意

    def retrieve_relevant_docs(query: str, top_k: int = 3) -> list:
        """检索相关文档片段"""
        # 简化实现，实际使用embedding相似度检索
        relevant = [doc for doc in knowledge_docs
                    if any(kw in doc['content'] for kw in query.split())]
        return relevant[:top_k]

    def answer_question(question: str) -> str:
        """回答风控相关问题"""
        relevant_docs = retrieve_relevant_docs(question)

        context = "\n\n".join([
            f"来源：{doc['title']}\n内容：{doc['content'][:1000]}"
            for doc in relevant_docs
        ])

        system_prompt = """
你是一位资深信贷风控专家，基于公司内部知识库回答问题。
- 只基于提供的文档内容回答，不要臆造
- 如文档中没有相关信息，明确说明
- 涉及数字、阈值等关键参数，必须引用来源文档
"""
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"参考文档：\n{context}\n\n问题：{question}"
            }]
        )
        return message.content[0].text

    return answer_question
```

---

## 15.5 LLM辅助风控模型开发

### 15.5.1 自动化EDA报告

```python
def generate_eda_narrative(eda_stats: dict, target_bad_rate: float) -> str:
    """
    将数值型EDA结果转化为可读报告
    """
    stats_str = json.dumps(eda_stats, ensure_ascii=False, indent=2)

    prompt = f"""
以下是一个信贷风控建模数据集的EDA统计结果，整体坏率为{target_bad_rate:.2%}。

EDA统计：
{stats_str}

请生成一份专业的EDA分析报告，包括：
1. 数据质量摘要（缺失、异常情况）
2. 关键特征的风险解读（哪些特征预测力强？为什么？）
3. 建模注意事项（需要特殊处理的特征）
4. 对特征工程的建议

使用中文，专业简洁，适合汇报给业务团队。
"""
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

### 15.5.2 模型监控异常解读

```python
def explain_monitoring_alert(alert_data: dict) -> str:
    """
    将监控告警数据转化为可理解的分析报告
    帮助算法工程师快速定位问题
    """
    prompt = f"""
以下是信贷风控模型的监控告警数据：

{json.dumps(alert_data, ensure_ascii=False, indent=2)}

请分析：
1. 告警的严重程度（紧急/一般/观察）
2. 可能的原因（数据问题/业务变化/模型过时）
3. 建议的排查步骤（按优先级排序）
4. 是否需要立即采取措施（如暂停模型/调整策略）
"""
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text
```

---

## 15.6 LLM应用的风险与合规

### 15.6.1 在风控场景使用LLM的注意事项

```markdown
⚠️ 使用LLM的核心风险：

1. 幻觉（Hallucination）
   - LLM可能生成看似合理但错误的数字/结论
   - 风控场景的数字精度要求高，所有LLM输出必须人工复核
   - 不得将LLM的输出直接用于核心信用决策

2. 数据安全
   - 客户个人信息（姓名、身份证、电话）不得传入外部LLM API
   - 必须数据脱敏或使用本地部署的模型（如Qwen、Llama等）

3. 可解释性
   - LLM的输出难以追溯，不满足监管要求的可解释性
   - LLM只能作为辅助工具，决策依据必须是传统模型

4. 一致性
   - LLM在相同输入下可能产生不同输出
   - 需要固定system prompt，并记录每次输出
```

### 15.6.2 本地模型部署（数据安全场景）

```python
# 使用本地模型（以Ollama为例）处理敏感数据
import requests

def local_llm_analyze(text: str, model: str = "qwen2.5:7b") -> str:
    """
    调用本地部署的LLM（无数据出境风险）
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": text,
            "stream": False,
        }
    )
    return response.json()["response"]
```

---

## 16.7 未来展望

```
LLM × 风控的演进方向：

近期（1~2年）：
├── LLM + RAG：企业尽调报告自动生成
├── LLM + 结构化数据：混合模型（文本+数值联合建模）
└── LLM + 催收：个性化话术，提升接触率

中期（2~3年）：
├── Agent × 风控：自动化模型监控与异常处理
├── 多模态风控：结合图像（营业执照识别）、语音（电话情绪分析）
└── AI辅助建模：从数据到模型的半自动化流水线

长期：
└── 实时自适应风控系统：模型自主发现模式、自主更新策略
    （仍需人工监督，监管要求是核心约束）
```

---

> **本章小结**：LLM在风控中是强大的辅助工具，而非核心决策引擎。其价值在于处理非结构化信息、提升人工效率、辅助知识管理。使用时必须严守数据安全红线（不传客户隐私数据），并保持清醒认识：核心信用决策的责任主体依然是经过严格验证的传统风控模型。
