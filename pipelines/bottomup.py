"""
Bottom-up 분석 파이프라인 v2 (고도화):
  Step 1. 비즈니스 모델    — 수익 구조·모델 유형·반복 매출 분석
  Step 2. 재무 건전성      — yfinance 실제 데이터 기반 PER·PBR·이익률·FCF 해석
  Step 3. 해자 및 경쟁력   — 경제적 해자 6가지 유형 평가
  Step 4. 리스크 요인      — 리스크 매트릭스 + 종합 투자 의견

모든 단계는 SystemMessage(페르소나/규칙) + HumanMessage(데이터/요청) 구조를 사용합니다.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts.templates import (
    BU_BUSINESS_MODEL_SYSTEM, BU_BUSINESS_MODEL_HUMAN,
    BU_FINANCIAL_HEALTH_SYSTEM, BU_FINANCIAL_HEALTH_HUMAN,
    BU_MOAT_SYSTEM,            BU_MOAT_HUMAN,
    BU_RISK_SYSTEM,            BU_RISK_HUMAN,
)
from tools.market_data import get_detailed_financials, format_detailed_financials_for_llm
from tools.news_search import search_news, build_search_queries


PIPELINE_STEPS = [
    {
        "label":  "🏢 Step 1/4: 비즈니스 모델 분석 중...",
        "key":    "step1_business",
        "system": BU_BUSINESS_MODEL_SYSTEM,
        "human":  BU_BUSINESS_MODEL_HUMAN,
        "needs_financial": True,
        "needs_news":      True,
    },
    {
        "label":  "💰 Step 2/4: 재무 건전성 분석 중...",
        "key":    "step2_financials",
        "system": BU_FINANCIAL_HEALTH_SYSTEM,
        "human":  BU_FINANCIAL_HEALTH_HUMAN,
        "needs_financial": True,
        "needs_news":      False,
    },
    {
        "label":  "🏰 Step 3/4: 해자 및 경쟁력 분석 중...",
        "key":    "step3_moat",
        "system": BU_MOAT_SYSTEM,
        "human":  BU_MOAT_HUMAN,
        "needs_financial": True,
        "needs_news":      True,
    },
    {
        "label":  "⚠️  Step 4/4: 리스크 요인 분석 중...",
        "key":    "step4_risks",
        "system": BU_RISK_SYSTEM,
        "human":  BU_RISK_HUMAN,
        "needs_financial": True,
        "needs_news":      True,
    },
]


def run_bottomup_pipeline(ticker: str, llm, competitors: list[str] | None = None) -> dict:
    """
    특정 기업에 대한 Bottom-up 분석(4단계)을 순차적으로 실행합니다.

    Args:
        ticker:      분석할 주식 티커 (예: TSLA, AAPL, 005930.KS)
        llm:         LangChain LLM 인스턴스
        competitors: 경쟁사 티커 목록 (현재 버전에서는 Step 3 컨텍스트로 활용)

    Returns:
        각 단계별 분석 결과 딕셔너리 + raw 재무 데이터
    """
    print(f"\n🔍 [{ticker}] Bottom-up 분석 시작 (4단계 전문 애널리스트 구조)...")

    # ── 1. yfinance 상세 재무 데이터 수집 ────────────────────────────────
    print("  📊 yfinance 재무 데이터 수집 중...")
    financial_raw  = get_detailed_financials(ticker)
    financial_text = format_detailed_financials_for_llm(financial_raw)
    company_name   = financial_raw.get("meta", {}).get("company_name", ticker)

    # 재무 데이터 딕셔너리를 콘솔에 출력 (검증용)
    _print_financial_snapshot(financial_raw)

    # ── 2. 뉴스 수집 ─────────────────────────────────────────────────────
    print("  📰 관련 뉴스 수집 중...")
    queries = build_search_queries(ticker, "company")
    news_parts = [search_news(q, max_results=3) for q in queries]
    combined_news = "\n\n---\n\n".join(news_parts)

    # ── 3. 4단계 순차 분석 ───────────────────────────────────────────────
    results = {
        "type":         "bottom-up",
        "ticker":       ticker,
        "company_name": company_name,
        "financial_raw": financial_raw,   # raw 데이터도 결과에 포함
    }

    for step in PIPELINE_STEPS:
        print(f"  {step['label']}")
        inputs = {
            "company":        company_name,
            "ticker":         ticker,
            "financial_data": financial_text if step["needs_financial"] else "",
            "news_content":   combined_news  if step["needs_news"]      else "",
        }
        results[step["key"]] = _run_analyst_step(
            system_prompt=step["system"],
            human_prompt=step["human"],
            llm=llm,
            inputs=inputs,
        )

    return results


def generate_bottomup_report(result: dict, generated_at: str) -> str:
    """
    Bottom-up 분석 결과를 최종 마크다운 보고서로 변환합니다.

    Args:
        result:       run_bottomup_pipeline()의 반환값
        generated_at: 분석 일시 문자열

    Returns:
        마크다운 형식의 완성된 보고서 문자열
    """
    ticker = result["ticker"]
    name   = result["company_name"]
    fr     = result.get("financial_raw", {})
    v      = fr.get("valuation", {})
    p      = fr.get("price", {})

    # 표지 요약 카드
    summary_card = f"""
> **시가총액** {v.get('market_cap', 'N/A')} | **현재가** ${p.get('current', 'N/A')} | \
**PER** {v.get('per_trailing', 'N/A')}배 | **PBR** {v.get('pbr', 'N/A')}배 | \
**52주 고/저** ${p.get('week52_high', 'N/A')} / ${p.get('week52_low', 'N/A')}
""".strip()

    report = f"""# 📊 기업 분석 보고서: {name} ({ticker})
> 분석 일시: {generated_at} | 분석 방법: Bottom-up v2 (비즈니스→재무→해자→리스크)

{summary_card}

---

{result['step1_business']}

---

{result['step2_financials']}

---

{result['step3_moat']}

---

{result['step4_risks']}

---
*⚠️ 본 보고서는 AI가 생성한 정보로, 투자 권유가 아닙니다. 투자 결정은 본인의 판단과 책임하에 이루어져야 합니다.*
"""
    return report.strip()


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

def _run_analyst_step(system_prompt: str, human_prompt: str, llm, inputs: dict) -> str:
    """SystemMessage + HumanMessage 구조로 단일 분석 단계를 실행합니다."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",  human_prompt),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs)


def _print_financial_snapshot(data: dict):
    """수집된 재무 데이터 핵심 지표를 콘솔에 출력합니다."""
    if "error" in data:
        print(f"  ⚠️  재무 데이터 수집 실패: {data['error']}")
        return
    v = data.get("valuation", {})
    p = data.get("price", {})
    pr = data.get("profitability", {})
    print(f"  ✅ 수집 완료 → "
          f"시총 {v.get('market_cap')} | "
          f"PER {v.get('per_trailing')}배 | "
          f"PBR {v.get('pbr')}배 | "
          f"현재가 ${p.get('current')} | "
          f"매출성장 {pr.get('revenue_growth')}")
