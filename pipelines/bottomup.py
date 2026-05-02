"""
Bottom-up 분석 파이프라인: 특정 기업/티커 입력 시 실행됩니다.
비즈니스 모델 → 재무 상태 → 경쟁사 비교 → 투자 리스크 순서로 분석합니다.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts.templates import (
    BUSINESS_MODEL_PROMPT,
    FINANCIAL_SUMMARY_PROMPT,
    COMPETITOR_ANALYSIS_PROMPT,
    RISK_ANALYSIS_PROMPT,
)
from tools.market_data import get_financial_summary, format_financial_for_llm
from tools.news_search import search_news, build_search_queries


def run_bottomup_pipeline(ticker: str, llm, competitors: list[str] | None = None) -> dict:
    """
    특정 기업에 대한 Bottom-up 분석을 순차적으로 실행합니다.

    Args:
        ticker: 분석할 기업의 주식 티커 (예: AAPL, 005930.KS)
        llm: LangChain LLM 인스턴스
        competitors: 비교할 경쟁사 티커 목록 (없으면 LLM이 판단)

    Returns:
        각 분석 단계별 결과를 담은 딕셔너리
    """
    print(f"\n🔍 [{ticker}] Bottom-up 분석 시작...")

    # ── Step 1: 데이터 수집 ────────────────────────
    print("  📈 재무 데이터 수집 중 (yfinance)...")
    financial_raw = get_financial_summary(ticker)
    financial_text = format_financial_for_llm(financial_raw)
    company_name = financial_raw.get("company_name", ticker)

    print("  📰 관련 뉴스 수집 중...")
    queries = build_search_queries(ticker, "company")
    news_parts = [search_news(q, max_results=3) for q in queries]
    combined_news = "\n\n---\n\n".join(news_parts)

    # ── Step 2: 비즈니스 모델 분석 ────────────────
    print("  🏢 Step 1/4: 비즈니스 모델 분석 중...")
    business_result = _run_step(BUSINESS_MODEL_PROMPT, llm, {
        "company": company_name,
        "news_content": combined_news,
    })

    # ── Step 3: 재무 상태 요약 ─────────────────────
    print("  💰 Step 2/4: 재무 상태 분석 중...")
    financial_result = _run_step(FINANCIAL_SUMMARY_PROMPT, llm, {
        "company": company_name,
        "financial_data": financial_text,
    })

    # ── Step 4: 경쟁사 비교 ───────────────────────
    print("  ⚔️  Step 3/4: 경쟁사 비교 분석 중...")
    competitor_context = _build_competitor_context(ticker, competitors, financial_text)
    competitor_result = _run_step(COMPETITOR_ANALYSIS_PROMPT, llm, {
        "company": company_name,
        "financial_data": competitor_context,
        "news_content": combined_news,
    })

    # ── Step 5: 투자 리스크 분석 ──────────────────
    print("  ⚠️  Step 4/4: 투자 리스크 분석 중...")
    risk_result = _run_step(RISK_ANALYSIS_PROMPT, llm, {
        "company": company_name,
        "financial_data": financial_text,
        "news_content": combined_news,
    })

    return {
        "type": "bottom-up",
        "ticker": ticker,
        "company_name": company_name,
        "step1_business": business_result,
        "step2_financials": financial_result,
        "step3_competitors": competitor_result,
        "step4_risks": risk_result,
    }


def _build_competitor_context(ticker: str, competitors: list | None, own_financial: str) -> str:
    """경쟁사 비교를 위한 컨텍스트를 구성합니다."""
    if not competitors:
        return f"[{ticker} 재무 데이터]\n{own_financial}\n\n(경쟁사 티커가 제공되지 않아 LLM이 알려진 경쟁사와 비교합니다)"

    from tools.market_data import get_financial_summary, format_financial_for_llm
    parts = [f"[{ticker} 재무 데이터]\n{own_financial}"]

    for comp in competitors[:3]:  # 최대 3개 경쟁사
        comp_data = get_financial_summary(comp)
        parts.append(f"\n[{comp} 재무 데이터]\n{format_financial_for_llm(comp_data)}")

    return "\n".join(parts)


def _run_step(template: str, llm, inputs: dict) -> str:
    """단일 분석 단계를 실행하고 결과를 반환합니다."""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs)
