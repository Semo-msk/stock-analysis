"""
yfinance를 사용해 주식 재무 데이터를 수집하는 모듈입니다.
"""

import yfinance as yf
import pandas as pd
from typing import Optional


def get_financial_summary(ticker: str) -> dict:
    """
    yfinance로 특정 기업의 최근 3년 재무 데이터를 가져옵니다.

    Args:
        ticker: 주식 티커 (예: AAPL, 005930.KS)

    Returns:
        재무 요약 딕셔너리
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # 연간 재무제표 (최근 3년)
        income_stmt = stock.financials          # 손익계산서
        balance_sheet = stock.balance_sheet    # 대차대조표

        summary = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": _format_number(info.get("marketCap")),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "revenue_growth": info.get("revenueGrowth"),
            "profit_margins": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
            "return_on_equity": info.get("returnOnEquity"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "description": info.get("longBusinessSummary", "")[:500],
        }

        # 최근 3년 매출 및 영업이익
        if not income_stmt.empty:
            revenue_row = _find_row(income_stmt, ["Total Revenue", "Revenue"])
            op_income_row = _find_row(income_stmt, ["Operating Income", "EBIT"])

            if revenue_row is not None:
                summary["revenue_3yr"] = {
                    str(col.year): _format_number(val)
                    for col, val in zip(income_stmt.columns[:3], revenue_row[:3])
                }
            if op_income_row is not None:
                summary["operating_income_3yr"] = {
                    str(col.year): _format_number(val)
                    for col, val in zip(income_stmt.columns[:3], op_income_row[:3])
                }

        # 부채비율 계산
        if not balance_sheet.empty:
            total_debt_row = _find_row(balance_sheet, ["Total Debt", "Long Term Debt"])
            equity_row = _find_row(balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest"])

            if total_debt_row is not None and equity_row is not None:
                debt = total_debt_row.iloc[0]
                equity = equity_row.iloc[0]
                if equity and equity != 0:
                    summary["debt_to_equity_calc"] = round((debt / equity) * 100, 1)

        return summary

    except Exception as e:
        return {"ticker": ticker, "error": str(e), "message": "데이터를 가져오는 데 실패했습니다."}


def get_competitor_data(ticker: str, competitors: Optional[list] = None) -> list[dict]:
    """
    경쟁사 재무 데이터를 수집합니다.

    Args:
        ticker: 분석 대상 티커
        competitors: 경쟁사 티커 목록 (없으면 빈 리스트 반환)

    Returns:
        경쟁사 데이터 리스트
    """
    results = []
    targets = [ticker] + (competitors or [])

    for t in targets:
        data = get_financial_summary(t)
        results.append({
            "ticker": t,
            "name": data.get("company_name", t),
            "market_cap": data.get("market_cap", "N/A"),
            "pe_ratio": data.get("pe_ratio", "N/A"),
            "profit_margins": _pct(data.get("profit_margins")),
            "revenue_growth": _pct(data.get("revenue_growth")),
            "return_on_equity": _pct(data.get("return_on_equity")),
        })

    return results


def format_financial_for_llm(financial_data: dict) -> str:
    """재무 데이터 딕셔너리를 LLM이 읽기 좋은 텍스트로 변환합니다."""
    if "error" in financial_data:
        return f"데이터 수집 실패: {financial_data['error']}"

    lines = [
        f"기업명: {financial_data.get('company_name')}",
        f"섹터/산업: {financial_data.get('sector')} / {financial_data.get('industry')}",
        f"시가총액: {financial_data.get('market_cap')}",
        f"현재 주가: {financial_data.get('current_price')}",
        f"PER (주가수익비율): {financial_data.get('pe_ratio')}",
        f"선행 PER: {financial_data.get('forward_pe')}",
        f"영업이익률: {_pct(financial_data.get('profit_margins'))}",
        f"매출 성장률: {_pct(financial_data.get('revenue_growth'))}",
        f"부채비율: {financial_data.get('debt_to_equity')}",
        f"자기자본이익률(ROE): {_pct(financial_data.get('return_on_equity'))}",
        f"52주 최고가: {financial_data.get('52_week_high')}",
        f"52주 최저가: {financial_data.get('52_week_low')}",
    ]

    if "revenue_3yr" in financial_data:
        lines.append("\n[최근 3년 매출]")
        for year, val in financial_data["revenue_3yr"].items():
            lines.append(f"  {year}년: {val}")

    if "operating_income_3yr" in financial_data:
        lines.append("\n[최근 3년 영업이익]")
        for year, val in financial_data["operating_income_3yr"].items():
            lines.append(f"  {year}년: {val}")

    if financial_data.get("description"):
        lines.append(f"\n[사업 설명]\n{financial_data['description']}")

    return "\n".join(lines)


# ── 내부 헬퍼 함수 ──────────────────────────────

def _find_row(df: pd.DataFrame, candidates: list) -> Optional[pd.Series]:
    """데이터프레임에서 후보 행 이름 중 존재하는 것을 반환합니다."""
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _format_number(value) -> str:
    """큰 숫자를 읽기 쉬운 단위로 변환합니다."""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if v >= 1e12:
            return f"${v/1e12:.2f}T"
        if v >= 1e9:
            return f"${v/1e9:.2f}B"
        if v >= 1e6:
            return f"${v/1e6:.2f}M"
        return str(v)
    except (TypeError, ValueError):
        return "N/A"


def _pct(value) -> str:
    """소수를 백분율 문자열로 변환합니다."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value)*100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"
