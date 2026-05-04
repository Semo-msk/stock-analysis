"""
사용자 입력을 'industry'(산업/테마) 또는 'company'(특정 기업)으로 분류하는 라우터입니다.
"""

import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts.templates import ROUTER_PROMPT


# 알려진 티커 패턴: 1~5개 대문자, 또는 한국 주식 코드 패턴
TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$|^\d{6}(\.KS|\.KQ)?$")

# 산업/테마 키워드
INDUSTRY_KEYWORDS = {
    "ai", "인공지능", "artificial intelligence", "반도체", "semiconductor",
    "자율주행", "autonomous", "ev", "전기차", "electric vehicle", "바이오",
    "biotechnology", "bio", "메타버스", "metaverse", "클라우드", "cloud",
    "핀테크", "fintech", "친환경", "renewable", "신재생", "로보틱스", "robotics",
    "우주", "space", "게임", "gaming", "사이버보안", "cybersecurity",
}

# 월간 리뷰 트리거 키워드 (--mode monthly 없이도 자동 감지)
MONTHLY_REVIEW_KEYWORDS = {
    "월간리뷰", "월간 리뷰", "모임자료", "모임 자료", "스터디자료",
    "monthly review", "market review", "월간시장", "이번달시장",
}


def classify_input(user_input: str, llm) -> dict:
    """
    사용자 입력을 분류합니다. 규칙 기반 분류 후 불확실하면 LLM에게 물어봅니다.

    Args:
        user_input: 사용자가 입력한 텍스트
        llm: LangChain LLM 인스턴스

    Returns:
        {"type": "industry"|"company"|"monthly_review", "normalized": str, "reason": str}
    """
    cleaned = user_input.strip()

    # 1단계: 규칙 기반 빠른 분류
    rule_result = _rule_based_classify(cleaned)
    if rule_result:
        return rule_result

    # 2단계: LLM 기반 분류
    return _llm_classify(cleaned, llm)


def _rule_based_classify(text: str) -> dict | None:
    """간단한 규칙으로 분류합니다. 확실하지 않으면 None 반환."""
    upper = text.upper().strip()
    lower = text.lower().strip()

    # 월간 리뷰 키워드 감지 (쉼표 구분 다중 테마 포함)
    for keyword in MONTHLY_REVIEW_KEYWORDS:
        if keyword in lower:
            return {
                "type": "monthly_review",
                "normalized": text,
                "reason": f"'{keyword}' 키워드가 감지된 월간 시장 리뷰 요청입니다."
            }

    # 티커 패턴 매칭 (예: AAPL, NVDA, 005930.KS)
    if TICKER_PATTERN.match(upper):
        return {
            "type": "company",
            "normalized": upper,
            "reason": f"'{text}'은 주식 티커 코드 형식입니다."
        }

    # 산업 키워드 매칭
    for keyword in INDUSTRY_KEYWORDS:
        if keyword in lower:
            return {
                "type": "industry",
                "normalized": text,
                "reason": f"'{keyword}' 키워드가 포함된 산업/테마 입력입니다."
            }

    return None  # 불확실 → LLM으로 넘김


def _llm_classify(text: str, llm) -> dict:
    """LLM을 사용해 입력을 분류합니다."""
    prompt = PromptTemplate.from_template(ROUTER_PROMPT)
    chain = prompt | llm | StrOutputParser()

    raw_output = chain.invoke({"user_input": text})

    # JSON 파싱
    try:
        # 마크다운 코드블록 제거 후 파싱
        json_text = re.sub(r"```json\s*|\s*```", "", raw_output).strip()
        result = json.loads(json_text)
        return result
    except json.JSONDecodeError:
        # 파싱 실패 시 기본값 반환
        return {
            "type": "industry",
            "normalized": text,
            "reason": "분류 불확실 - 산업/테마로 기본 처리합니다."
        }
