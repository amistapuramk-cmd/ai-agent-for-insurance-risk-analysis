"""
AI Agent for Insurance Risk Analysis
------------------------------------
A simple base project with two layers:

1. Risk engine  - transparent, rule-based scoring (works offline).
2. AI agent     - Claude uses the engine as a tool, reasons over the result
                  and writes an underwriting memo (needs ANTHROPIC_API_KEY).

Usage:
    python risk_agent.py --demo              # run sample applicants
    python risk_agent.py --file applicant.json
    python risk_agent.py --demo --no-llm     # engine only, no API calls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import Any

# --------------------------------------------------------------------------
# 1. DATA MODEL
# --------------------------------------------------------------------------


@dataclass
class Applicant:
    name: str
    age: int
    policy_type: str = "health"          # health | life | auto
    coverage_amount: float = 100_000     # sum insured
    annual_income: float = 50_000
    smoker: bool = False
    bmi: float = 24.0
    chronic_conditions: int = 0          # count of diagnosed conditions
    claims_last_5y: int = 0
    traffic_violations: int = 0          # mainly relevant for auto
    occupation_hazard: int = 1           # 1 (desk job) .. 5 (very hazardous)
    late_payments: int = 0               # missed/late premium payments


@dataclass
class RiskResult:
    applicant: str
    score: float
    tier: str
    decision: str
    premium_multiplier: float
    estimated_annual_premium: float
    factors: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------
# 2. RISK ENGINE
# --------------------------------------------------------------------------

BASE_RATES = {"health": 0.020, "life": 0.008, "auto": 0.030}  # % of coverage

GUIDELINES = """\
Underwriting guidelines (v1):
- Score 0-24   : Low risk       -> approve at standard rate
- Score 25-49  : Moderate risk  -> approve with premium loading
- Score 50-74  : High risk      -> refer to a human underwriter
- Score 75-100 : Very high risk -> decline or request extra medical/financial evidence
- Any decision must be explained using the listed risk factors.
- Final decisions on High / Very High risk always need human review.
"""


def assess_risk(a: Applicant) -> RiskResult:
    """Score an applicant from 0 (safe) to 100 (very risky)."""
    factors: list[tuple[str, float, str]] = []

    def add(name: str, pts: float, why: str) -> None:
        if pts:
            factors.append((name, pts, why))

    # Age
    if a.age < 25:
        add("Age", 8 if a.policy_type == "auto" else 3, f"Young applicant ({a.age})")
    elif a.age > 60:
        add("Age", 12 if a.policy_type in ("health", "life") else 6, f"Older applicant ({a.age})")
    elif a.age > 45:
        add("Age", 5, f"Middle-age band ({a.age})")

    # Health-related (health & life)
    if a.policy_type in ("health", "life"):
        if a.smoker:
            add("Smoking", 15, "Smoker")
        if a.bmi >= 35:
            add("BMI", 15, f"BMI {a.bmi} (severe obesity)")
        elif a.bmi >= 30:
            add("BMI", 10, f"BMI {a.bmi} (obese)")
        elif a.bmi < 18.5:
            add("BMI", 6, f"BMI {a.bmi} (underweight)")
        add("Chronic conditions", min(a.chronic_conditions * 5, 15),
            f"{a.chronic_conditions} chronic condition(s)")

    # Driving (auto)
    if a.policy_type == "auto":
        add("Traffic violations", min(a.traffic_violations * 5, 20),
            f"{a.traffic_violations} violation(s)")

    # Claims history
    add("Claims history", min(a.claims_last_5y * 6, 24),
        f"{a.claims_last_5y} claim(s) in last 5 years")

    # Occupation
    add("Occupation", (a.occupation_hazard - 1) * 3,
        f"Occupation hazard level {a.occupation_hazard}/5")

    # Financial exposure
    ratio = a.coverage_amount / max(a.annual_income, 1)
    if ratio > 20:
        add("Coverage vs income", 14, f"Coverage is {ratio:.0f}x annual income")
    elif ratio > 10:
        add("Coverage vs income", 8, f"Coverage is {ratio:.0f}x annual income")

    # Payment behaviour
    add("Payment history", min(a.late_payments * 3, 9),
        f"{a.late_payments} late payment(s)")

    score = min(sum(p for _, p, _ in factors), 100)

    if score < 25:
        tier, decision = "Low", "APPROVE (standard rate)"
    elif score < 50:
        tier, decision = "Moderate", "APPROVE with premium loading"
    elif score < 75:
        tier, decision = "High", "REFER to human underwriter"
    else:
        tier, decision = "Very High", "DECLINE / request more evidence"

    multiplier = round(1 + (score / 100) * 1.5, 2)
    premium = round(a.coverage_amount * BASE_RATES.get(a.policy_type, 0.02) * multiplier, 2)

    return RiskResult(
        applicant=a.name,
        score=round(score, 1),
        tier=tier,
        decision=decision,
        premium_multiplier=multiplier,
        estimated_annual_premium=premium,
        factors=[{"factor": n, "points": p, "detail": w}
                 for n, p, w in sorted(factors, key=lambda f: -f[1])],
    )


# --------------------------------------------------------------------------
# 3. AI AGENT (Claude with tool use)
# --------------------------------------------------------------------------

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are an insurance underwriting assistant.
For every applicant: call `assess_applicant` to get the risk score, and call
`get_guidelines` to check company policy. Then write a short underwriting memo with:
1) Summary  2) Key risk drivers  3) Recommendation  4) Suggested next steps.
Never invent numbers - use only tool results. You advise; humans make final decisions."""

TOOLS = [
    {
        "name": "assess_applicant",
        "description": "Run the rule-based risk engine on an applicant and return score, tier, decision, premium and factors.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "policy_type": {"type": "string", "enum": ["health", "life", "auto"]},
                "coverage_amount": {"type": "number"},
                "annual_income": {"type": "number"},
                "smoker": {"type": "boolean"},
                "bmi": {"type": "number"},
                "chronic_conditions": {"type": "integer"},
                "claims_last_5y": {"type": "integer"},
                "traffic_violations": {"type": "integer"},
                "occupation_hazard": {"type": "integer", "minimum": 1, "maximum": 5},
                "late_payments": {"type": "integer"},
            },
            "required": ["name", "age"],
        },
    },
    {
        "name": "get_guidelines",
        "description": "Return the company's underwriting guidelines.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def run_tool(name: str, args: dict[str, Any]) -> str:
    if name == "assess_applicant":
        return json.dumps(asdict(assess_risk(Applicant(**args))))
    if name == "get_guidelines":
        return GUIDELINES
    return f"Unknown tool: {name}"


def agent_memo(applicant: Applicant, max_steps: int = 6) -> str:
    """Let Claude analyse the applicant, calling tools as needed."""
    import anthropic  # imported here so the engine works without the SDK

    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [
        {"role": "user",
         "content": "Analyse this insurance applicant:\n" + json.dumps(asdict(applicant), indent=2)}
    ]

    for _ in range(max_steps):
        resp = client.messages.create(
            model=MODEL, max_tokens=1200, system=SYSTEM_PROMPT,
            tools=TOOLS, messages=messages,
        )
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text")

        messages.append({"role": "assistant", "content": resp.content})
        results = [
            {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
            for b in resp.content if b.type == "tool_use"
        ]
        messages.append({"role": "user", "content": results})

    return "Agent stopped: too many steps."


# --------------------------------------------------------------------------
# 4. CLI
# --------------------------------------------------------------------------

DEMO_APPLICANTS = [
    Applicant("Asha Rao", 29, "health", 200_000, 60_000, False, 22.5, 0, 0, 0, 1, 0),
    Applicant("Ravi Kumar", 52, "life", 500_000, 40_000, True, 33.0, 2, 1, 0, 3, 1),
    Applicant("Neha Singh", 22, "auto", 30_000, 25_000, False, 23.0, 0, 2, 3, 2, 2),
]


def print_result(r: RiskResult) -> None:
    print(f"\n=== {r.applicant} ===")
    print(f"Score: {r.score}/100 | Tier: {r.tier} | Decision: {r.decision}")
    print(f"Premium multiplier: x{r.premium_multiplier} | Est. annual premium: {r.estimated_annual_premium:,.2f}")
    for f in r.factors:
        print(f"  +{f['points']:>4}  {f['factor']}: {f['detail']}")


def main() -> None:
    p = argparse.ArgumentParser(description="AI Agent for Insurance Risk Analysis")
    p.add_argument("--demo", action="store_true", help="run built-in sample applicants")
    p.add_argument("--file", help="JSON file with one applicant object or a list")
    p.add_argument("--no-llm", action="store_true", help="skip the Claude agent")
    args = p.parse_args()

    if args.file:
        with open(args.file) as fh:
            data = json.load(fh)
        applicants = [Applicant(**d) for d in (data if isinstance(data, list) else [data])]
    elif args.demo:
        applicants = DEMO_APPLICANTS
    else:
        p.print_help()
        sys.exit(0)

    use_llm = not args.no_llm and bool(os.getenv("ANTHROPIC_API_KEY"))
    if not args.no_llm and not use_llm:
        print("(ANTHROPIC_API_KEY not set - running engine only)")

    for a in applicants:
        print_result(assess_risk(a))
        if use_llm:
            print("\n--- AI Underwriting Memo ---")
            print(agent_memo(a))


if __name__ == "__main__":
    main()
