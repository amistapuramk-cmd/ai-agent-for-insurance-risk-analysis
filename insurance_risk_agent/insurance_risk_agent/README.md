# AI Agent for Insurance Risk Analysis

Rule-based risk engine + Claude agent that writes underwriting memos.

## Setup
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=your_key   # optional; without it only the engine runs

## Run
    python risk_agent.py --demo
    python risk_agent.py --demo --no-llm
    python risk_agent.py --file applicant.json

## Extend
- Add new factors in `assess_risk()`
- Add new tools to `TOOLS` / `run_tool()` (e.g. fraud check, claims DB lookup)
- Wrap in Streamlit or FastAPI for a UI/API
