"""
DuckDuckGo(무료) 또는 Tavily API로 뉴스/웹 정보를 수집하는 모듈입니다.
환경변수 TAVILY_API_KEY가 있으면 Tavily를, 없으면 DuckDuckGo를 사용합니다.
"""

import os
from typing import Literal


def search_news(query: str, max_results: int = 5) -> str:
    """
    주어진 키워드로 뉴스/웹 정보를 검색하고 텍스트로 반환합니다.

    Args:
        query: 검색 키워드
        max_results: 가져올 결과 수

    Returns:
        검색 결과를 합친 텍스트
    """
    tavily_key = os.getenv("TAVILY_API_KEY")

    if tavily_key:
        return _search_with_tavily(query, max_results, tavily_key)
    else:
        return _search_with_duckduckgo(query, max_results)


def _search_with_tavily(query: str, max_results: int, api_key: str) -> str:
    """Tavily API로 검색합니다 (더 정확한 뉴스 검색)."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
        )

        parts = []
        if response.get("answer"):
            parts.append(f"[요약 답변]\n{response['answer']}\n")

        for i, result in enumerate(response.get("results", []), 1):
            parts.append(
                f"[뉴스 {i}] {result.get('title', '')}\n"
                f"출처: {result.get('url', '')}\n"
                f"내용: {result.get('content', '')[:400]}\n"
            )

        return "\n".join(parts) if parts else "검색 결과가 없습니다."

    except Exception as e:
        return f"Tavily 검색 실패: {e}\nDuckDuckGo로 재시도합니다."


def _search_with_duckduckgo(query: str, max_results: int) -> str:
    """DuckDuckGo로 검색합니다 (무료, API 키 불필요)."""
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"[뉴스] {r.get('title', '')}\n"
                    f"출처: {r.get('href', '')}\n"
                    f"내용: {r.get('body', '')[:400]}\n"
                )

        return "\n".join(results) if results else "검색 결과가 없습니다."

    except Exception as e:
        return f"DuckDuckGo 검색 실패: {e}"


def build_search_queries(topic: str, analysis_type: Literal["industry", "company"]) -> list[str]:
    """
    분석 유형에 맞는 검색 쿼리 목록을 생성합니다.

    Args:
        topic: 산업명 또는 기업명/티커
        analysis_type: "industry" 또는 "company"

    Returns:
        검색 쿼리 리스트
    """
    if analysis_type == "industry":
        return [
            f"{topic} industry market trends 2024 2025",
            f"{topic} sector growth forecast investment",
            f"{topic} leading companies stock outlook",
        ]
    else:
        return [
            f"{topic} company latest news 2024 2025",
            f"{topic} stock analysis business model revenue",
            f"{topic} competitors market share comparison",
            f"{topic} investment risks opportunities",
        ]
