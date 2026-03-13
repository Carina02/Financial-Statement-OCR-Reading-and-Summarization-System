# 财务报表自动化解析与审计 Agent (PoC Demo)

基于 LangGraph 构建的自动化财务报表处理管道。本系统通过状态机（State Machine）控制流，集成了低成本前置文档截断、微软高精度 OCR 版面分析以及大模型数据自修正循环，用于提取、校验并结构化非标财务 PDF 数据。

## 架构特性

* **低成本漏斗设计**：使用本地轻量级库 (`PyMuPDF`) 结合端侧/轻量级 LLM (`gpt-4o-mini`) 进行首页行业分类校验。拦截非目标客户数据，避免无效调用昂贵的云端 OCR 接口。
* **确定性版面还原**：采用 Azure AI Document Intelligence (`prebuilt-layout`)，将复杂表格精准映射为 Markdown 格式，消除大模型直接读取 PDF 的视觉幻觉。
* **闭环自修正机制 (Self-Correction)**：在发现财务等式（如资产负债表配平）存在逻辑断层时，系统会自动将错误上下文抛回给 LLM 进行修正，最多重试 3 次，超过阈值则触发熔断并转交人工审核。
* **强类型输出约束**：通过 Pydantic 约束 LLM 结构化输出（Structured Output），确保最终导出的字典列表与 Excel 表格零格式异常。

## 环境依赖与配置

本脚本基于 Python 3.10+ 测试通过。

### 1. 安装核心依赖包
```bash
pip install langgraph langchain langchain-openai langchain-community azure-ai-documentintelligence pymupdf pandas openpyxl pydantic

# OpenAI 凭证 (用于逻辑推理与自修正)
export OPENAI_API_KEY="sk-..."

# Azure Document Intelligence 凭证 (用于复杂版面与表格提取)
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://<your-resource-name>[.cognitiveservices.azure.com/](https://.cognitiveservices.azure.com/)"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="<your-azure-key>"

