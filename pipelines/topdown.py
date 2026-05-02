"""
Top-down 분석 파이프라인 v2 (고도화):
  Step 1. 거시 환경 (Macro)      — 금리·인플레이션 등 산업 유불리
  Step 2. 산업 사이클             — 도입기/성장기/성숙기/쇠퇴기 진단
  Step 3. 최근 산업 동향          — 지난 6개월 뉴스 스크랩 및 시그널 분류
  Step 4. 핵심 동인 (Catalyst)   — 향후 1~2년 주가 촉매제
  Step 5. Top Picks (최선호주)   — 글로벌 2 + 국내 1 대장주 선정

각 단계는 SystemMessage(페르소나/규칙) + HumanMessage(데이터/요청) 구조를 사용합니다.
LangChain의 ChatPromptTemplate.from_messages() 로 구성되어 역할 분리가 명확합니다.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts.templates import (
    MACRO_ENV_SYSTEM,      MACRO_ENV_HUMAN,
    INDUSTRY_CYCLE_SYSTEM, INDUSTRY_CYCLE_HUMAN,
    RECENT_TRENDS_SYSTEM,  RECENT_TRENDS_HUMAN,
    CATALYST_SYSTEM,       CATALYST_HUMAN,
    TOP_PICKS_SYSTEM,      TOP_PICKS_HUMAN,
)
from tools.news_search import search_news, build_search_queries


# 각 단계의 (시스템 프롬프트, 휴먼 프롬프트) 설정
PIPELINE_STEPS = [
    {
        "label": "📊 Step 1/5: 거시 환경 (Macro) 분석 중...",
        "key": "step1_macro",
        "system": MACRO_ENV_SYSTEM,
        "human": MACRO_ENV_HUMAN,
    },
    {
        "label": "🔄 Step 2/5: 산업 사이클 진단 중...",
        "key": "step2_cycle",
        "system": INDUSTRY_CYCLE_SYSTEM,
        "human": INDUSTRY_CYCLE_HUMAN,
    },
    {
        "label": "📰 Step 3/5: 최근 산업 동향 분석 중...",
        "key": "step3_trends",
        "system": RECENT_TRENDS_SYSTEM,
        "human": RECENT_TRENDS_HUMAN,
    },
    {
        "label": "🚀 Step 4/5: 핵심 동인 (Catalyst) 분석 중...",
        "key": "step4_catalyst",
        "system": CATALYST_SYSTEM,
        "human": CATALYST_HUMAN,
    },
    {
        "label": "🏆 Step 5/5: Top Picks (최선호주) 선정 중...",
        "key": "step5_picks",
        "system": TOP_PICKS_SYSTEM,
        "human": TOP_PICKS_HUMAN,
    },
]


def run_topdown_pipeline(theme: str, llm) -> dict:
    """
    산업/테마에 대한 고도화된 Top-down 분석(5단계)을 순차적으로 실행합니다.

    Args:
        theme: 분석할 산업/테마 (예: "AI 반도체", "자율주행")
        llm: LangChain LLM 인스턴스 (ChatGoogleGenerativeAI, ChatAnthropic 등)

    Returns:
        각 단계별 분석 결과를 담은 딕셔너리
    """
    print(f"\n🔍 [{theme}] Top-down 분석 시작 (5단계 전문 애널리스트 구조)...")

    # ── 뉴스 수집: 일반 + 최근 6개월 트렌드 전용 쿼리 ─────────────────────
    print("  📡 관련 뉴스 및 정보 수집 중...")
    general_queries = build_search_queries(theme, "industry")
    trend_queries = [
        f"{theme} industry news 2025",
        f"{theme} market trends recent months analysis",
        f"{theme} policy regulation government support 2024 2025",
    ]

    general_news = "\n\n---\n\n".join(
        search_news(q, max_results=3) for q in general_queries
    )
    trend_news = "\n\n---\n\n".join(
        search_news(q, max_results=3) for q in trend_queries
    )
    # Step 3(동향)은 최신 뉴스 전용, 나머지는 general 사용
    news_by_step = {
        "step1_macro":    general_news,
        "step2_cycle":    general_news,
        "step3_trends":   trend_news,
        "step4_catalyst": general_news + "\n\n" + trend_news,
        "step5_picks":    general_news,
    }

    # ── 5단계 순차 실행 ────────────────────────────────────────────────────
    results = {"type": "top-down", "theme": theme}

    for step in PIPELINE_STEPS:
        print(f"  {step['label']}")
        results[step["key"]] = _run_analyst_step(
            system_prompt=step["system"],
            human_prompt=step["human"],
            llm=llm,
            inputs={
                "theme": theme,
                "news_content": news_by_step[step["key"]],
            },
        )

    return results


def _run_analyst_step(system_prompt: str, human_prompt: str, llm, inputs: dict) -> str:
    """
    SystemMessage + HumanMessage 구조로 단일 분석 단계를 실행합니다.

    일반 PromptTemplate 대비 장점:
    - system 역할로 LLM의 페르소나와 출력 규칙을 고정
    - human 역할로 실제 분석 데이터를 전달해 역할 혼재 방지
    - 모델이 지시를 더 정확히 따르고 형식 준수율이 높아짐
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(inputs)
