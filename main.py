"""
주식투자 분석 자동화 시스템 - 메인 진입점

사용법:
    python main.py
    python main.py --input "NVDA"
    python main.py --input "AI 반도체" --output report.md
    python main.py --input "AAPL" --competitors "MSFT,GOOGL"
"""

import argparse
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def build_llm():
    """
    환경변수에 따라 Gemini 또는 Claude LLM을 초기화합니다.
    GOOGLE_API_KEY가 있으면 Gemini, ANTHROPIC_API_KEY가 있으면 Claude를 사용합니다.
    """
    google_key = os.getenv("GOOGLE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("✅ LLM: Google Gemini 사용")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_key,
            temperature=0.3,
        )
    elif anthropic_key:
        from langchain_anthropic import ChatAnthropic
        print("✅ LLM: Anthropic Claude 사용")
        return ChatAnthropic(
            model="claude-sonnet-4-6",
            anthropic_api_key=anthropic_key,
            temperature=0.3,
        )
    else:
        print("❌ 오류: GOOGLE_API_KEY 또는 ANTHROPIC_API_KEY 환경변수를 설정해 주세요.")
        print("   .env 파일에 다음 중 하나를 추가하세요:")
        print("   GOOGLE_API_KEY=your_key_here")
        print("   ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)


def format_report(result: dict) -> str:
    """분석 결과를 읽기 좋은 마크다운 보고서로 변환합니다."""
    now = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")

    if result["type"] == "top-down":
        title = f"# 📈 산업/테마 분석 보고서: {result['theme']}"
        body = f"""
{title}
> 분석 일시: {now} | 분석 방법: Top-down v2 (거시→사이클→동향→촉매→최선호주)

---

{result['step1_macro']}

---

{result['step2_cycle']}

---

{result['step3_trends']}

---

{result['step4_catalyst']}

---

{result['step5_picks']}

---
*⚠️ 본 보고서는 AI가 생성한 정보로, 투자 권유가 아닙니다. 투자 결정은 본인의 판단과 책임하에 이루어져야 합니다.*
"""
    else:
        title = f"# 📊 기업 분석 보고서: {result['company_name']} ({result['ticker']})"
        body = f"""
{title}
> 분석 일시: {now} | 분석 방법: Bottom-up (기업→재무→경쟁→리스크)

---

{result['step1_business']}

---

{result['step2_financials']}

---

{result['step3_competitors']}

---

{result['step4_risks']}

---
*⚠️ 본 보고서는 AI가 생성한 정보로, 투자 권유가 아닙니다. 투자 결정은 본인의 판단과 책임하에 이루어져야 합니다.*
"""
    return body.strip()


def save_report(content: str, output_path: str):
    """분석 보고서를 파일로 저장합니다."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ 보고서가 저장되었습니다: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="주식투자 분석 자동화 시스템")
    parser.add_argument("--input", "-i", type=str, help="분석 대상 (예: AAPL, AI 반도체)")
    parser.add_argument("--output", "-o", type=str, help="보고서 저장 경로 (예: report.md)")
    parser.add_argument("--competitors", "-c", type=str, help="경쟁사 티커 (쉼표 구분, 예: MSFT,GOOGL)")
    args = parser.parse_args()

    print("=" * 60)
    print("  📊 주식투자 분석 자동화 시스템")
    print("=" * 60)

    # 입력 받기
    user_input = args.input
    if not user_input:
        user_input = input("\n분석할 주제를 입력하세요\n(예: AI반도체 / NVDA / 자율주행 / AAPL)\n> ").strip()

    if not user_input:
        print("입력값이 없습니다. 프로그램을 종료합니다.")
        sys.exit(0)

    # LLM 초기화
    llm = build_llm()

    # 입력 분류 (라우팅)
    print(f"\n🤖 입력 분류 중: '{user_input}'")
    from router import classify_input
    classification = classify_input(user_input, llm)

    print(f"  → 분류 결과: {classification['type'].upper()}")
    print(f"  → 정제된 입력: {classification['normalized']}")
    print(f"  → 판단 이유: {classification['reason']}")

    # 파이프라인 실행
    result = None
    if classification["type"] == "industry":
        from pipelines.topdown import run_topdown_pipeline
        result = run_topdown_pipeline(classification["normalized"], llm)

    elif classification["type"] == "company":
        from pipelines.bottomup import run_bottomup_pipeline
        competitors = args.competitors.split(",") if args.competitors else None
        result = run_bottomup_pipeline(classification["normalized"], llm, competitors)

    # 보고서 생성
    if result:
        report = format_report(result)
        print("\n" + "=" * 60)
        print(report)

        # 파일 저장
        output_path = args.output
        if not output_path:
            safe_name = classification["normalized"].replace(" ", "_").replace("/", "-")
            output_path = f"report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"

        save_report(report, output_path)


if __name__ == "__main__":
    main()
