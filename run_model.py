import os
import fitz  # PyMuPDF
import pandas as pd
from typing import TypedDict, Optional, Dict, Literal, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader


# 1. 结构化输出定义 (强制 LLM 的输出格式)
class FinancialAnalysis(BaseModel):
    is_valid: bool = Field(description="资产负债表等核心财务逻辑是否配平、无严重缺失。")
    error_reason: str = Field(
        description="如果 is_valid 为 False，详细指出具体的错误原因和错位数据；如果为 True，输出空字符串。")
    summary: str = Field(description="财务数据的简单总结。")
    table_data: List[Dict[str, str]] = Field(description="提取的表格数据字典列表。")


# 2. Agent 全局状态定义
class GraphState(TypedDict):
    file_path: str
    industry: Optional[str]
    ocr_markdown: Optional[str]
    analysis_result: Optional[FinancialAnalysis]
    correction_attempts: int
    needs_excel: bool


# 初始化模型 (区分轻量与重型以控制成本)
llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_heavy = ChatOpenAI(model="gpt-4o", temperature=0)
# 绑定结构化输出
structured_llm = llm_heavy.with_structured_output(FinancialAnalysis)


# ==================== 节点 (Nodes) ====================

def node_classify_industry(state: GraphState):
    """前置低成本校验：提取首页文本判断行业"""
    print("[Agent] 正在进行低成本行业核验...")
    doc = fitz.open(state["file_path"])
    # 仅提取前 1000 字符，速度极快且零 API 成本
    first_page_text = doc[0].get_text()[:1000]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个行业分类器。阅读文本，判断其属于哪个行业。仅输出 'A', 'B', 'C' 或 'Other'。")
    ])
    industry = (prompt | llm_mini).invoke({"text": first_page_text}).content.strip()
    return {"industry": industry, "correction_attempts": 0}


def node_azure_ocr(state: GraphState):
    """高精度 OCR 提取 (仅当行业校验通过后执行)"""
    print("[Agent] 行业符合，调用 Azure Document Intelligence 提取版面...")
    loader = AzureAIDocumentIntelligenceLoader(
        api_endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
        api_key=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
        file_path=state["file_path"],
        api_model="prebuilt-layout",
        mode="markdown"
    )
    docs = loader.load()
    return {"ocr_markdown": docs[0].page_content}


def node_logic_check(state: GraphState):
    """闭卷语义逻辑检查与结构化抽取"""
    print(f"[Agent] 执行财务逻辑核查 (第 {state['correction_attempts']} 次修正后)...")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个高级财务审计。检查以下 Markdown 格式的财务报表。\n1. 严格核对资产负债是否配平。\n2. 检查是否有明显的数字错位或逻辑断层。\n如果发现错误，设置 is_valid=False 并说明原因。如果逻辑正常，设置 is_valid=True 并提取总结与表格。"),
        ("human", "{markdown_text}")
    ])
    result = (prompt | structured_llm).invoke({"markdown_text": state["ocr_markdown"]})
    return {"analysis_result": result}


def node_self_correction(state: GraphState):
    """LLM 数据自修正"""
    print(f"[Agent] 发现逻辑错误：{state['analysis_result'].error_reason}。正在尝试自我修正 OCR 文本...")
    prompt_text = f"""
    原 OCR 文本存在以下逻辑错误或错位：{state['analysis_result'].error_reason}

    原始 Markdown 文本：
    {state['ocr_markdown']}

    请你根据财务常识和上下文，修正 Markdown 中的错位数据或修复错误，直接返回修正后的 Markdown 全文，不要有任何多余的废话。
    """
    fixed_markdown = llm_heavy.invoke(prompt_text).content
    # 修正次数 +1
    return {"ocr_markdown": fixed_markdown, "correction_attempts": state["correction_attempts"] + 1}


def node_complexity_check(state: GraphState):
    """判断是否需要制作 Excel"""
    print("[Agent] 逻辑检查通过。评估数据复杂度...")
    table_data = state["analysis_result"].table_data
    # 简单启发式策略：如果提取到的表格字典列表长度大于 0，就认为需要 Excel
    needs_excel = len(table_data) > 0
    return {"needs_excel": needs_excel}


def node_export_excel(state: GraphState):
    """导出 Excel"""
    print("[Agent] 数据包含复杂表格，正在生成 Excel...")
    df = pd.DataFrame(state["analysis_result"].table_data)
    df.to_excel("financial_report_output.xlsx", index=False)
    print("Excel 生成完毕：financial_report_output.xlsx")
    return state


# ==================== 路由 (Conditional Edges) ====================

def route_after_classification(state: GraphState) -> Literal["node_azure_ocr", "END"]:
    if state["industry"] in ["A", "B", "C"]:
        return "node_azure_ocr"
    print(f"终止：当前行业为 '{state['industry']}'，不属于目标客户行业。")
    return "END"


def route_after_logic_check(state: GraphState) -> Literal["node_self_correction", "node_complexity_check"]:
    if not state["analysis_result"].is_valid:
        if state["correction_attempts"] < 3:
            return "node_self_correction"
        else:
            # 超过3次失败，直接向外抛出异常转交人工
            raise Exception(
                f"🚨 致命错误：LLM 尝试自我修正 3 次后仍未解决财务逻辑错误。最后一次错误原因：{state['analysis_result'].error_reason}。请转交人工审核！")
    return "node_complexity_check"


def route_after_complexity(state: GraphState) -> Literal["node_export_excel", "END"]:
    if state["needs_excel"]:
        return "node_export_excel"
    print("报表结构简单，无需生成 Excel。")
    return "END"


# ==================== 构建图与编译 ====================

workflow = StateGraph(GraphState)

workflow.add_node("node_classify", node_classify_industry)
workflow.add_node("node_azure_ocr", node_azure_ocr)
workflow.add_node("node_logic", node_logic_check)
workflow.add_node("node_self_correction", node_self_correction)
workflow.add_node("node_complexity", node_complexity_check)
workflow.add_node("node_export_excel", node_export_excel)

workflow.add_edge(START, "node_classify")
workflow.add_conditional_edges("node_classify", route_after_classification,
                               {"node_azure_ocr": "node_azure_ocr", "END": END})
workflow.add_edge("node_azure_ocr", "node_logic")
workflow.add_conditional_edges("node_logic", route_after_logic_check, {"node_self_correction": "node_self_correction",
                                                                       "node_complexity_check": "node_complexity"})
workflow.add_edge("node_self_correction", "node_logic")
workflow.add_conditional_edges("node_complexity", route_after_complexity,
                               {"node_export_excel": "node_export_excel", "END": END})
workflow.add_edge("node_export_excel", END)

app = workflow.compile()

if __name__ == "__main__":
    # 测试运行入口
    inputs = {"file_path": "demo_report.pdf"}
    try:
        for output in app.stream(inputs):
            pass  # 进度已在 Node 内部 Print
        print("\n 工作流执行完毕。")
    except Exception as e:
        print(f"\n捕获到系统异常: {e}")