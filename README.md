# Automated Financial Report Parsing and Auditing Agent (PoC Demo)

An automated financial report processing pipeline built on LangGraph. This system utilizes a state machine control flow, integrating a low-cost front-end document truncation, Microsoft's high-precision OCR layout analysis, and an LLM data self-correction loop to extract, validate, and structure non-standard financial PDF data.

## Architecture Features

* **Low-Cost Funnel Design**: Utilizes a local lightweight library (`PyMuPDF`) combined with a lightweight LLM (`gpt-4o-mini`) for first-page industry classification validation. It intercepts non-target client data to avoid invalid and expensive calls to cloud OCR APIs.
* **Deterministic Layout Restoration**: Employs Azure AI Document Intelligence (`prebuilt-layout`) to accurately map complex tables into Markdown format, eliminating the visual hallucinations typically caused by LLMs directly reading PDFs.
* **Closed-Loop Self-Correction Mechanism**: Upon detecting logical gaps in financial equations (e.g., balance sheet imbalances), the system automatically feeds the error context back to the LLM for correction. It retries up to 3 times before triggering a circuit breaker and transferring the task to human review.
* **Strongly Typed Output Constraints**: Uses Pydantic to enforce LLM Structured Output, ensuring zero formatting exceptions in the final exported dictionary list and Excel spreadsheet.

## Environment Dependencies & Configuration

This script has been tested and verified on Python 3.10+.

### 1. Install Core Dependencies
```bash
pip install langgraph langchain langchain-openai langchain-community azure-ai-documentintelligence pymupdf pandas openpyxl pydantic
```

### 2. Configure Environment Variables

The execution of this system strictly depends on the following API keys. Please export them in your terminal environment beforehand:
```bash
# OpenAI Credentials (for logic reasoning and self-correction)
export OPENAI_API_KEY="sk-..."

# Azure Document Intelligence Credentials (for complex layout and table extraction)
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://<your-resource-name>[.cognitiveservices.azure.com/](https://.cognitiveservices.azure.com/)"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="<your-azure-key>"
```

## Core Workflow (LangGraph Nodes)

### The system strictly follows these nodes for state transitions:

1. `node_classify_industry`: Reads the first 1000 characters of the PDF. If the industry does not belong to [A, B, C], it executes `END` to terminate the process.

2. `node_azure_ocr`: Once industry validation passes, it calls Azure to extract the Markdown of the entire PDF.

3. `node_logic_check`: Conducts a closed-book audit of the financial logic. If is_valid=False, it routes to the self-correction node.

4. `node_self_correction`: Corrects the Markdown based on the error reasons from the previous node, then returns to `node_logic_check`. The maximum retry threshold is set to 3.

5. `node_complexity_check`: Evaluates the length of the extracted table array to determine the subsequent output strategy.

6. `node_export_excel`: Cleans and exports highly complex report data into a `.xlsx` format.

## Usage Instructions

Place the target test file (e.g., `demo_report.pdf`) in the project root directory and modify the input path in the main program entry point:
```bash
if __name__ == "__main__":
    inputs = {"file_path": "demo_report.pdf"}
    for output in app.stream(inputs):
        pass
```
Execute the script:
```bash
python main.py
```

Expected Output
* Terminal Logs: Prints the state machine transitions and self-correction interception logs.

* File Artifacts: Upon successful execution, `financial_report_output.xlsx` will be generated in the same directory.

## Future fix
The following modules shall be refactored:

* File Storage Mechanism: The Demo currently uses local paths (file_path). A production environment requires integration with object storage like AWS S3 / Alibaba Cloud OSS.

* Asynchronous Execution: Azure OCR parsing of a multi-page PDF may take 10-30 seconds. In production, this Agent must be encapsulated into an asynchronous task queue (e.g., Celery + Redis) to prevent blocking the main thread.

* Human-in-the-loop: When the system throws an Exception (after 3 failed self-correction attempts), the data of the current State must be dumped to the client's internal review frontend for manual correction and return.
