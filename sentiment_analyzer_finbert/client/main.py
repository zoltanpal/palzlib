from sentiment_analyzer_finbert import SentimentAnalyzer


def main() -> None:
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_text("Sports bets on prediction markets ruled to be swaps, exempt from state laws")
    print("result:", result.asdict())


if __name__ == "__main__":
    main()
