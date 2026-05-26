import asyncio
import urllib.request
import urllib.parse
import json
import time
import os

from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.types import TemplatedSystemInstructions

# --- 1. Custom Tools for the Agent ---

def fetch_europe_pmc_batch(query: str, max_results: int = 5) -> str:
    """Fetches a batch of scholarly papers from Europe PMC based on the query.
    
    Args:
        query: The search query string (e.g., '(ACL OR ACLR) AND IMU').
        max_results: Maximum number of papers to return in this batch.
    """
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": max_results
    }
    
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        results = data.get("resultList", {}).get("result", [])
        papers = []
        for res in results:
            if not res.get("abstractText"):
                continue
            paper = {
                "title": res.get("title", ""),
                "authors": res.get("authorString", ""),
                "year": res.get("pubYear", ""),
                "journal": res.get("journalTitle", ""),
                "url": f"https://doi.org/{res.get('doi')}" if res.get('doi') else "",
                "citations": res.get("citedByCount", 0),
                "abstract": res.get("abstractText", "")
            }
            papers.append(paper)
        
        return json.dumps(papers, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error fetching papers: {str(e)}"

def append_to_markdown_table(markdown_row: str) -> str:
    """Appends a correctly formatted Markdown table row to the candidates file.
    
    Args:
        markdown_row: A single string representing a Markdown table row. MUST be formatted with | delimiters and contain all 15 required columns.
    """
    file_path = "mds/agent_acl_imu_gait_candidates.md"
    
    # Initialize file with header if it doesn't exist
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# ACL IMU Gait Analysis - Agent Reviewed Candidates\n\n")
            f.write("| 제목 | Research Question | 사용 데이터 | 데이터 규모 | 환자 수 | 속도 조건 | 수술 후 개월 수 | 분석 방법 | 사용한 모델 | 핵심 결과(메트릭 포함) | 연구 한계 | 출처 | 링크 | 논문 게재지 | 인용수 | 실제 발췌문 (영문/국문 번역) | 심층 탐색 여부 및 근거 |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    with open(file_path, "a", encoding="utf-8") as f:
        # Ensure row ends with newline
        if not markdown_row.endswith("\n"):
            markdown_row += "\n"
        f.write(markdown_row)
    
    return "Row successfully appended to markdown table."

# --- 2. Agent Setup & Execution ---

async def run_research_agent():
    print("Starting Automated Literature Review Agent...")
    
    # Define Persona and System Instructions
    system_instructions = TemplatedSystemInstructions(
        identity=(
            "You are an expert sports medicine and machine learning research assistant. "
            "Your task is to analyze academic abstracts provided in JSON format, extract specific metadata, "
            "translate actual English excerpts into Korean, and write the data into a Markdown table using a tool."
        ),
        mandates=[
            "For each paper in the input, you MUST extract: Research Question, Data Size, Patient Count, Speed Condition, Months Post-Op, Analysis Method, Model used, Key Results, Limitations.",
            "You MUST extract the first 2-3 sentences of the abstract as the 'actual excerpt' in English.",
            "You MUST accurately translate that 'actual excerpt' into Korean.",
            "You MUST format the combined excerpt as: **[EN]** <english text><br>**[KR]** <korean text>.",
            "For EACH paper, call the `append_to_markdown_table` tool exactly once, passing the fully formatted row.",
            "The row must contain 17 columns matching the header: 제목 | Research Question | 사용 데이터 | 데이터 규모 | 환자 수 | 속도 조건 | 수술 후 개월 수 | 분석 방법 | 사용한 모델 | 핵심 결과 | 연구 한계 | 출처(Year) | 링크 | 논문 게재지 | 인용수 | 실제 발췌문 (영/한) | 심층 탐색 근거."
        ]
    )
    
    config = LocalAgentConfig(
        tools=[fetch_europe_pmc_batch, append_to_markdown_table],
        system_instructions=system_instructions
    )
    
    async with Agent(config) as agent:
        print("Agent initialized. Instructing agent to fetch and process papers...")
        
        prompt = (
            "Please use the `fetch_europe_pmc_batch` tool to search for papers using the query: "
            "'(\"anterior cruciate ligament\" OR ACL OR ACLR) AND (\"inertial measurement unit\" OR IMU OR \"wearable sensor\") AND (gait OR walking) AND (classification OR regression)'. "
            "Fetch a batch of 3 papers. Then, carefully read their abstracts, extract all required fields, translate the excerpts into Korean, "
            "and use the `append_to_markdown_table` tool to record EACH paper as a Markdown row."
        )
        
        response = await agent.chat(prompt)
        
        # Stream the reasoning/thoughts and the final response
        print("\n--- Agent's Thought Process & Actions ---")
        async for chunk in response:
            print(chunk, end="", flush=True)
        print("\n--- Finished ---")

if __name__ == "__main__":
    # Ensure the script is run in an environment with the GEMINI_API_KEY set.
    if "GEMINI_API_KEY" not in os.environ:
        print("ERROR: GEMINI_API_KEY environment variable is missing.")
        print("Please set it before running this script (e.g. export GEMINI_API_KEY='your_key').")
    else:
        asyncio.run(run_research_agent())
