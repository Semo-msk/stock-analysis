"""
yfinance를 사용해 주식 재무 데이터를 수집하는 모듈입니다.
"""

import yfinance as yf
import pandas as pd
from typing import Optional


# ── 공개 API ───────────────────────────────────────────────────────────────

def get_detailed_financials(ticker: str) -> dict:
    """
    Bottom-up 분석에 필요한 상세 재무 데이터를 딕셔너리로 반환합니다.

    반환 구조:
      meta        : 기업 기본 정보
      valuation   : PER·PBR·EV/EBITDA 등 밸류에이션 지표
      profitability: 최근 3년 매출·영업이익률·순이익률
      health      : 부채비율·유동비율·이자보상배율
      cashflow    : FCF·FCF 수익률
      price       : 현재가·52주 고저·베타
    """
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        inc   = stock.financials       # 손익계산서 (연간)
        bal   = stock.balance_sheet    # 대차대조표 (연간)
        cf    = stock.cashflow         # 현금흐름표 (연간)

        data: dict = {}

        # ── 기본 정보 ──────────────────────────────
        data["meta"] = {
            "ticker":       ticker,
            "company_name": info.get("longName", ticker),
            "sector":       info.get("sector", "N/A"),
            "industry":     info.get("industry", "N/A"),
            "country":      info.get("country", "N/A"),
            "employees":    info.get("fullTimeEmployees"),
            "description":  (info.get("longBusinessSummary") or "")[:600],
        }

        # ── 밸류에이션 ─────────────────────────────
        data["valuation"] = {
            "market_cap":       _fmt(info.get("marketCap")),
            "per_trailing":     _round(info.get("trailingPE")),     # 현재 PER
            "per_forward":      _round(info.get("forwardPE")),      # 선행 PER
            "pbr":              _round(info.get("priceToBook")),     # PBR
            "ev_ebitda":        _round(info.get("enterpriseToEbitda")),
            "ps_ratio":         _round(info.get("priceToSalesTrailing12Months")),
        }

        # ── 수익성: 최근 3년 매출·영업이익·순이익 ───
        data["profitability"] = _extract_profitability(inc, info)

        # ── 재무 건전성 ────────────────────────────
        data["health"] = _extract_health(bal, inc, info)

        # ── 현금흐름 (FCF) ─────────────────────────
        data["cashflow"] = _extract_cashflow(cf, info)

        # ── 주가 정보 ──────────────────────────────
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        data["price"] = {
            "current":          current_price,
            "week52_high":      info.get("fiftyTwoWeekHigh"),
            "week52_low":       info.get("fiftyTwoWeekLow"),
            "beta":             _round(info.get("beta")),
            "target_mean":      info.get("targetMeanPrice"),
            "analyst_rating":   info.get("recommendationKey", "N/A"),
        }

        return data

    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def format_detailed_financials_for_llm(data: dict) -> str:
    """
    get_detailed_financials() 결과를 LLM에게 전달할 구조화된 텍스트로 변환합니다.
    초보자도 읽을 수 있도록 각 지표에 간단한 설명을 괄호로 추가합니다.
    """
    if "error" in data:
        return f"⚠️ 데이터 수집 실패: {data['error']}"

    m  = data.get("meta", {})
    v  = data.get("valuation", {})
    pr = data.get("profitability", {})
    h  = data.get("health", {})
    c  = data.get("cashflow", {})
    p  = data.get("price", {})

    lines = [
        "=" * 55,
        f"  {m.get('company_name')} ({m.get('ticker')})",
        f"  섹터: {m.get('sector')} | 국가: {m.get('country')}",
        "=" * 55,

        "\n[밸류에이션 — 주가가 비싼지 싼지 판단하는 지표]",
        f"  시가총액               : {v.get('market_cap')}",
        f"  PER-현재 (주가수익비율): {v.get('per_trailing')}배  ← 낮을수록 저평가 가능",
        f"  PER-선행 (미래 실적 기준): {v.get('per_forward')}배",
        f"  PBR (주가순자산비율)   : {v.get('pbr')}배  ← 1배 미만이면 자산 대비 싼 편",
        f"  EV/EBITDA              : {v.get('ev_ebitda')}배",
        f"  PSR (주가매출비율)     : {v.get('ps_ratio')}배",
    ]

    # 수익성 테이블
    lines.append("\n[수익성 — 돈을 잘 버는지 보는 지표 (최근 3개 회계연도)]")
    rev  = pr.get("revenue", {})
    op_m = pr.get("op_margin", {})
    net_m = pr.get("net_margin", {})
    if rev:
        lines.append(f"  {'연도':<8} {'매출':<14} {'영업이익률':<12} {'순이익률'}")
        lines.append(f"  {'-'*50}")
        for yr in sorted(rev.keys(), reverse=True):
            lines.append(
                f"  {yr:<8} {rev.get(yr,'N/A'):<14} "
                f"{op_m.get(yr,'N/A'):<12} {net_m.get(yr,'N/A')}"
            )
    lines += [
        f"  매출 성장률 (YoY)      : {pr.get('revenue_growth')}",
        f"  ROE (자기자본이익률)   : {pr.get('roe')}  ← 높을수록 주주 돈을 잘 활용",
        f"  ROA (총자산이익률)     : {pr.get('roa')}",
    ]

    # 재무 건전성
    lines += [
        "\n[재무 건전성 — 빚이 얼마나 있는지, 갚을 능력이 있는지]",
        f"  부채비율               : {h.get('debt_to_equity')}  ← 100% 미만 양호, 200% 이상 주의",
        f"  유동비율 (단기 상환 능력): {h.get('current_ratio')}  ← 1.5 이상이면 안전",
        f"  이자보상배율           : {h.get('interest_coverage')}  ← 3 이상이면 이자 갚기 충분",
        f"  총부채                 : {h.get('total_debt')}",
        f"  현금 및 현금성 자산    : {h.get('cash')}",
    ]

    # 현금흐름
    lines += [
        "\n[현금흐름 — 실제로 손에 쥐는 현금]",
        f"  잉여현금흐름 FCF       : {c.get('fcf')}  ← 양수면 현금 창출 능력 우수",
        f"  FCF 수익률             : {c.get('fcf_yield')}  ← 높을수록 현금 창출 효율 좋음",
        f"  영업활동현금흐름       : {c.get('operating_cf')}",
    ]

    # 주가 정보
    lines += [
        "\n[주가 정보]",
        f"  현재가                 : ${p.get('current')}",
        f"  52주 최고가 / 최저가   : ${p.get('week52_high')} / ${p.get('week52_low')}",
        f"  베타 (시장 민감도)     : {p.get('beta')}  ← 1 이상이면 시장보다 변동성 큼",
        f"  애널리스트 목표가      : ${p.get('target_mean')}",
        f"  애널리스트 투자 의견   : {p.get('analyst_rating')}",
    ]

    if m.get("description"):
        lines += [f"\n[사업 개요]\n  {m['description']}"]

    return "\n".join(lines)


# ── 하위 호환: 기존 함수 유지 ──────────────────────────────────────────────

def get_financial_summary(ticker: str) -> dict:
    """get_detailed_financials()의 하위 호환 래퍼 (flat dict 반환)."""
    data = get_detailed_financials(ticker)
    if "error" in data:
        return data
    flat = {"ticker": ticker}
    flat.update(data.get("meta", {}))
    flat.update(data.get("valuation", {}))
    flat.update(data.get("price", {}))
    pr = data.get("profitability", {})
    flat["revenue_growth"] = pr.get("revenue_growth_raw")
    flat["profit_margins"] = pr.get("net_margin_raw")
    flat["return_on_equity"] = pr.get("roe_raw")
    h = data.get("health", {})
    flat["debt_to_equity"] = h.get("debt_to_equity_raw")
    flat["revenue_3yr"]         = data.get("profitability", {}).get("revenue", {})
    flat["operating_income_3yr"] = data.get("profitability", {}).get("op_income", {})
    return flat


def format_financial_for_llm(financial_data: dict) -> str:
    """하위 호환용: flat dict를 받아 텍스트로 변환합니다."""
    if "error" in financial_data:
        return f"데이터 수집 실패: {financial_data['error']}"
    ticker = financial_data.get("ticker", "")
    detailed = get_detailed_financials(ticker)
    if "error" in detailed:
        return f"데이터 수집 실패: {detailed['error']}"
    return format_detailed_financials_for_llm(detailed)


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

def _extract_profitability(inc: pd.DataFrame, info: dict) -> dict:
    """손익계산서에서 3년 매출·이익률을 추출합니다."""
    result: dict = {
        "revenue": {},
        "op_income": {},
        "op_margin": {},
        "net_margin": {},
        "revenue_growth":     _pct(info.get("revenueGrowth")),
        "revenue_growth_raw": info.get("revenueGrowth"),
        "roe":     _pct(info.get("returnOnEquity")),
        "roe_raw": info.get("returnOnEquity"),
        "roa":     _pct(info.get("returnOnAssets")),
        "net_margin_raw": info.get("profitMargins"),
    }

    if inc.empty:
        return result

    rev_row = _find_row(inc, ["Total Revenue", "Revenue"])
    op_row  = _find_row(inc, ["Operating Income", "EBIT"])
    net_row = _find_row(inc, ["Net Income", "Net Income Common Stockholders"])

    cols = inc.columns[:3]
    for col in cols:
        yr = str(col.year)
        rev = rev_row[col] if rev_row is not None else None
        op  = op_row[col]  if op_row  is not None else None
        net = net_row[col] if net_row is not None else None

        result["revenue"][yr]   = _fmt(rev)
        result["op_income"][yr] = _fmt(op)
        result["op_margin"][yr] = _pct_from_ratio(op, rev)
        result["net_margin"][yr] = _pct_from_ratio(net, rev)

    return result


def _extract_health(bal: pd.DataFrame, inc: pd.DataFrame, info: dict) -> dict:
    """대차대조표에서 건전성 지표를 추출합니다."""
    result: dict = {
        "debt_to_equity":     f"{info.get('debtToEquity', 'N/A')}",
        "debt_to_equity_raw": info.get("debtToEquity"),
        "current_ratio":      _round(info.get("currentRatio")),
        "total_debt":         _fmt(info.get("totalDebt")),
        "cash":               _fmt(info.get("totalCash")),
        "interest_coverage":  "N/A",
    }

    # 이자보상배율 = 영업이익 / 이자비용
    if not bal.empty and not inc.empty:
        op_row       = _find_row(inc, ["Operating Income", "EBIT"])
        interest_row = _find_row(inc, ["Interest Expense", "Interest Expense Non Operating"])
        if op_row is not None and interest_row is not None:
            op_val  = op_row.iloc[0]
            int_val = abs(interest_row.iloc[0]) if interest_row.iloc[0] else None
            if int_val and int_val != 0:
                result["interest_coverage"] = f"{op_val / int_val:.1f}배"

    return result


def _extract_cashflow(cf: pd.DataFrame, info: dict) -> dict:
    """현금흐름표에서 FCF를 추출합니다."""
    result: dict = {
        "fcf":          "N/A",
        "fcf_yield":    "N/A",
        "operating_cf": "N/A",
    }
    if cf.empty:
        return result

    op_cf_row  = _find_row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
    capex_row  = _find_row(cf, ["Capital Expenditure", "Purchase Of Property Plant And Equipment"])

    if op_cf_row is not None:
        op_cf = op_cf_row.iloc[0]
        result["operating_cf"] = _fmt(op_cf)

        if capex_row is not None:
            capex = abs(capex_row.iloc[0])
            fcf_val = op_cf - capex
            result["fcf"] = _fmt(fcf_val)

            market_cap = info.get("marketCap")
            if market_cap and market_cap != 0:
                result["fcf_yield"] = _pct(fcf_val / market_cap)

    return result


def _find_row(df: pd.DataFrame, candidates: list) -> Optional[pd.Series]:
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        v = float(value)
        if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
        if abs(v) >= 1e9:  return f"${v/1e9:.2f}B"
        if abs(v) >= 1e6:  return f"${v/1e6:.2f}M"
        return f"{v:.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _round(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _pct(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value)*100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _pct_from_ratio(numerator, denominator) -> str:
    try:
        if numerator is None or denominator is None or float(denominator) == 0:
            return "N/A"
        return f"{float(numerator)/float(denominator)*100:.1f}%"
    except (TypeError, ValueError, ZeroDivisionError):
        return "N/A"
