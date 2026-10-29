from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment_vader(df):
    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    sentiment_types = []

    for msg in df['message']:
        score = analyzer.polarity_scores(str(msg))['compound']
        sentiments.append(score)

        if score >= 0.05:
            sentiment_types.append("Positive")
        elif score <= -0.05:
            sentiment_types.append("Negative")
        else:
            sentiment_types.append("Neutral")

    df['sentiment'] = sentiments
    df['sentiment_type'] = sentiment_types
    return df
