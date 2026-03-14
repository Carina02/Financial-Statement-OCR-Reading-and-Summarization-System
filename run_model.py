import os
import fitz  # PyMuPDF
import pandas as pd
from typing import TypedDict, Optional, Dict, Literal, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

# 1. Structured output definition (Force LLM output format)
class FinancialAnalysis(BaseModel):
    is_valid: bool = Field(description="Whether core financial logic such as the balance sheet is balanced and free of severe omissions.")
    error_reason: str = Field(description="If is_valid is False, detail the specific error reasons and misaligned data; if True, output an empty string.")
    summary: str = Field(description="A simple summary of the financial data.")
    table_data: List[Dict[str, str]] = Field(description="List of extracted table data dictionaries.")

# 2. Global state definition for the Agent
class GraphState(TypedDict):
    file_path: str
    industry: Optional[str]
    ocr_markdown: Optional[str]
    analysis_result: Optional[FinancialAnalysis]
    correction_attempts: int
    needs_excel: bool

# Initialize models (Separate lightweight and heavy models to control costs)
llm_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_heavy = ChatOpenAI(model="gpt-4o", temperature=0)

# Bind structured output
structured_llm = llm_heavy.with_structured_output(FinancialAnalysis)

# ==================== Nodes ====================

def node_classify_industry(state: GraphState):
    """Preliminary low-cost validation: Extract first-page text to determine the industry."""
    print("[Agent] Performing low-cost industry validation...")
    doc = fitz.open(state["file_path"])
    # Extract only the first 1000 characters, extremely fast with zero API cost
    first_page_text = doc[0].get_text()[:1000] 
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an industry classifier. Read the text and determine its industry. Output ONLY 'A', 'B', 'C', or 'Other'.")
    ])
    industry = (prompt | llm_mini).invoke({"text": first_page_text}).content.strip()
    return {"industry": industry, "correction_attempts": 0}

def node_azure_ocr(state: GraphState):
    """High-precision OCR extraction (Executed only if industry validation passes)."""
    print("[Agent] Industry matched. Calling Azure Document Intelligence for layout extraction...")
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
    """Closed-book semantic logic check and structured extraction."""
    print(f"[Agent] Executing financial logic check (After correction attempt {state['correction_attempts']})...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior financial auditor. Check the following financial report in Markdown format.\n1. Strictly verify if the balance sheet is balanced.\n2. Check for obvious numerical misalignments or logical gaps.\nIf errors are found, set is_valid=False and explain the reason. If the logic is normal, set is_valid=True and extract the summary and tables."),
        ("human", "{markdown_text}")
    ])
    result = (prompt | structured_llm).invoke({"markdown_text": state["ocr_markdown"]})
    return {"analysis_result": result}

def node_self_correction(state: GraphState):
    """LLM Data Self-Correction."""
    print(f"[Agent] Logic error detected: {state['analysis_result'].error_reason}. Attempting to self-correct the OCR text...")
    prompt_text = f"""
    The original OCR text has the following logic errors or misalignments: {state['analysis_result'].error_reason}
    
    Original Markdown text:
    {state['ocr_markdown']}
    
    Please correct the misaligned data or fix the errors in the Markdown based on financial common sense and context. Return ONLY the fully corrected Markdown text without any extra nonsense.
    """
    fixed_markdown = llm_heavy.invoke(prompt_text).content
    # Increment correction attempts
    return {"ocr_markdown": fixed_markdown, "correction_attempts": state["correction_attempts"] + 1}

def node_complexity_check(state: GraphState):
    """Determine if an Excel file needs to be generated."""
    print("[Agent] Logic check passed. Evaluating data complexity...")
    table_data = state["analysis_result"].table_data
    # Simple heuristic: If the extracted table dictionary list length is > 0, consider Excel necessary.
    needs_excel = len(table_data) > 0 
    return {"needs_excel": needs_excel}

def node_export_excel(state: GraphState):
    """Export to Excel."""
    print("[Agent] Data contains complex tables, generating Excel...")
    df = pd.DataFrame(state["analysis_result"].table_data)
    df.to_excel("financial_report_output.xlsx", index=False)
    print("✅ Excel generation completed: financial_report_output.xlsx")
    return state

# ==================== Routing (Conditional Edges) ====================

def route_after_classification(state: GraphState) -> Literal["node_azure_ocr", "END"]:
    if state["industry"] in ["A", "B", "C"]:
        return "node_azure_ocr"
    print(f"❌ Terminated: Current industry is '{state['industry']}', which does not belong to the target client industries.")
    return "END"

def route_after_logic_check(state: GraphState) -> Literal["node_self_correction", "node_complexity_check"]:
    if not state["analysis_result"].is_valid:
        if state["correction_attempts"] < 3:
            return "node_self_correction"
        else:
            # Exceeded 3 failed attempts, throw an exception directly for manual review
            raise Exception(f"🚨 Fatal Error: LLM failed to resolve the financial logic error after 3 self-correction attempts. Last error reason: {state['analysis_result'].error_reason}. Please transfer to human review!")
    return "node_complexity_check"

def route_after_complexity(state: GraphState) -> Literal["node_export_excel", "END"]:
    if state["needs_excel"]:
        return "node_export_excel"
    print("ℹ️ Report structure is simple, no need to generate Excel.")
    return "END"

# ==================== Build and Compile Graph ====================

workflow = StateGraph(GraphState)

workflow.add_node("node_classify", node_classify_industry)
workflow.add_node("node_azure_ocr", node_azure_ocr)
workflow.add_node("node_logic", node_logic_check)
workflow.add_node("node_self_correction", node_self_correction)
workflow.add_node("node_complexity", node_complexity_check)
workflow.add_node("node_export_excel", node_export_excel)

workflow.add_edge(START, "node_classify")
workflow.add_conditional_edges("node_classify", route_after_classification, {"node_azure_ocr": "node_azure_ocr", "END": END})
workflow.add_edge("node_azure_ocr", "node_logic")
workflow.add_conditional_edges("node_logic", route_after_logic_check, {"node_self_correction": "node_self_correction", "node_complexity_check": "node_complexity"})
workflow.add_edge("node_self_correction", "node_logic")
workflow.add_conditional_edges("node_complexity", route_after_complexity, {"node_export_excel": "node_export_excel", "END": END})
workflow.add_edge("node_export_excel", END)

app = workflow.compile()

if __name__ == "__main__":
    # Test execution entry point
    inputs = {"file_path": "demo_report.pdf"}
    try:
        for output in app.stream(inputs):
            pass # Progress is already printed inside Nodes
        print("\n🎉 Workflow execution completed.")
    except Exception as e:
        print(f"\nCaught system exception: {e}")
