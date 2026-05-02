"""
Top-down 분석 파이프라인: 산업/테마 입력 시 실행됩니다.
거시경제 → 산업 성장성 → 핵심 수혜 기업 순서로 분석합니다.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts.templates import (
    MACRO_ANALYSIS_PROMPT,
    INDUSTRY_GROWTH_PROMPT,
    KEY_PLAYERS_PROMPT,
)
from tools.news_search import search_news, build_search_queries


def run_topdown_pipeline(theme: str, llm) -> dict:
    """
    산업/테마에 대한 Top-down 분석을 순차적으로 실행합니다.

    Args:
        theme: 분석할 산업/테마 (예: "AI 반도체", "자율주행")
        llm: LangChain LLM 인스턴스

    Returns:
        각 분석 단계별 결과를 담은 딕셔너리
    """
    print(f"\n🔍 [{theme}] Top-down 분석 시작...")
    queries = build_search_queries(theme, "industry")

    # ── Step 1: 뉴스 수집 ─────────────────────────
    print("  📰 관련 뉴스 및 정보 수집 중...")
    news_parts = [search_news(q, max_results=3) for q in queries]
    combined_news = "\n\n---\n\n".join(news_parts)

    # ── Step 2: 거시경제 분석 ──────────────────────
    print("  📊 Step 1/3: 거시경제 동향 분석 중...")
    macro_result = _run_step(MACRO_ANALYSIS_PROMPT, llm, {
        "theme": theme,
        "news_content": combined_news,
    })

    # ── Step 3: 산업 성장성 분석 ──────────────────
    print("  🚀 Step 2/3: 산업 성장성 분석 중...")
    growth_result = _run_step(INDUSTRY_GROWTH_PROMPT, llm, {
        "theme": theme,
        "news_content": combined_news,
    })

    # ── Step 4: 핵심 수혜 기업 도출 ───────────────
    print("  🏆 Step 3/3: 핵심 수혜 기업 분석 중...")
    players_result = _run_step(KEY_PLAYERS_PROMPT, llm, {
        "theme": theme,
        "news_content": combined_news,
    })

    return {
        "type": "top-down",
        "theme": theme,
        "step1_macro": macro_result,
        "step2_growth": growth_result,
        "step3_players": players_result,
    }


def _run_step(template: str, llm, inputs: dict) -> str:
    """단일 분석 단계를 실행하고 결과를 반환합니다."""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs)
