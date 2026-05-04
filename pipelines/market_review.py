"""
Monthly Market Review 파이프라인:
  Step 1. 시장 헤드라인  — 이번 달 시장을 한 마디로 (핵심 변수 + 현황)
  Step 2. 테마별 분석    — 각 테마를 "왜/최근/뷰" 3-question 구조로 분석
  Step 3. 토론 + 일정   — 토론 질문 2~3개 + 다음 모니터링 일정표

사용 예:
  python main.py --input "이란전쟁,스페이스X IPO,루멘텀홀딩스" --mode monthly
"""

from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts.templates import (
    MR_HEADLINE_SYSTEM,   MR_HEADLINE_HUMAN,
    MR_THEME_SYSTEM,      MR_THEME_HUMAN,
    MR_DISCUSSION_SYSTEM, MR_DISCUSSION_HUMAN,
)
from tools.news_search import search_news, build_search_queries


_ORDINALS = {1: "첫 번째", 2: "두 번째", 3: "세 번째", 4: "네 번째", 5: "다섯 번째"}
_EMOJIS   = {1: "1️⃣", 2: "2️⃣", 3: "3️⃣", 4: "4️⃣", 5: "5️⃣"}


def run_market_review_pipeline(themes: list[str], llm, date: str = None) -> dict:
    """
    월간 시장 리뷰를 3단계로 생성합니다.

    Args:
        themes: 분석할 테마 목록 (예: ["이란전쟁", "스페이스X IPO", "루멘텀홀딩스"])
        llm:    LangChain LLM 인스턴스
        date:   기준 날짜 문자열 (기본: 오늘)

    Returns:
        headline, theme_sections, discussion 을 담은 딕셔너리
    """
    if not date:
        date = datetime.now().strftime("%Y년 %m월 %d일")

    print(f"\n🔍 월간 시장 리뷰 시작 — {len(themes)}개 테마: {', '.join(themes)}")

    # ── 1. 테마별 뉴스 수집 ────────────────────────────────────────────────────
    print("  📡 테마별 뉴스 수집 중...")
    theme_news: dict[str, str] = {}
    for theme in themes:
        queries = build_search_queries(theme, "industry")
        parts   = [search_news(q, max_results=3) for q in queries]
        theme_news[theme] = "\n\n---\n\n".join(parts)

    all_news = "\n\n===\n\n".join(
        f"[{t}]\n{n}" for t, n in theme_news.items()
    )

    # ── 2. 시장 헤드라인 ───────────────────────────────────────────────────────
    print("  🌐 Step 1: 시장 헤드라인 작성 중...")
    headline = _run_step(
        MR_HEADLINE_SYSTEM, MR_HEADLINE_HUMAN, llm,
        {
            "date":         date,
            "themes":       ", ".join(themes),
            "news_content": all_news,
        },
    )

    # ── 3. 테마별 상세 분석 ────────────────────────────────────────────────────
    theme_sections: list[str] = []
    for i, theme in enumerate(themes):
        idx = i + 1
        print(f"  📊 Step 2-{idx}/{len(themes)}: [{theme}] 분석 중...")
        section = _run_step(
            MR_THEME_SYSTEM, MR_THEME_HUMAN, llm,
            {
                "theme":         theme,
                "theme_index":   str(idx),
                "theme_emoji":   _EMOJIS.get(idx, f"{idx}️⃣"),
                "theme_ordinal": _ORDINALS.get(idx, f"{idx}번째"),
                "date":          date,
                "news_content":  theme_news[theme],
            },
        )
        theme_sections.append(section)

    # ── 4. 토론 질문 + 모니터링 일정 ──────────────────────────────────────────
    print("  💬 Step 3: 토론 질문 + 모니터링 일정 작성 중...")
    discussion = _run_step(
        MR_DISCUSSION_SYSTEM, MR_DISCUSSION_HUMAN, llm,
        {
            "date":         date,
            "themes":       ", ".join(themes),
            "news_content": all_news,
        },
    )

    return {
        "type":           "market-review",
        "date":           date,
        "themes":         themes,
        "headline":       headline,
        "theme_sections": theme_sections,
        "discussion":     discussion,
    }


def generate_market_review_report(result: dict, generated_at: str) -> str:
    """
    run_market_review_pipeline() 결과를 최종 마크다운 보고서로 변환합니다.
    """
    date   = result["date"]
    themes = result["themes"]

    sections_text = "\n\n".join(result["theme_sections"])
    keywords      = " ".join(f"`{t}`" for t in themes)

    positions_template = """\
## 📝 내 포지션 변화 (선택 · 공개 가능한 범위에서)

| 구분 | 종목/ETF | 이유 |
|---|---|---|
| 매수 | | |
| 매도 | | |
| 비중 확대 | | |
| 비중 축소 | | |"""

    report = f"""\
# 📈 월간 시장 리뷰 — {date}
> 분석 일시: {generated_at} | 주요 테마: {", ".join(themes)}

---

{result["headline"]}

## 📊 주목하고 있는 시장 / 테마

{sections_text}

---

{result["discussion"]}

{positions_template}

---

## ⭐ 이번 달 시장 키워드

{keywords}

---
*⚠️ 본 보고서는 AI가 생성한 정보로, 투자 권유가 아닙니다. 투자 결정은 본인의 판단과 책임하에 이루어져야 합니다.*"""

    return report.strip()


def _run_step(system_prompt: str, human_prompt: str, llm, inputs: dict) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",  human_prompt),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs)
