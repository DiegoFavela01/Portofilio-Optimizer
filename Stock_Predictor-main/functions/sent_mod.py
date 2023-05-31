import requests
import pandas as pd
import nltk
import datetime as dt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os

# Turn off warning signs for cleaner code
from warnings import filterwarnings
filterwarnings("ignore")

def market_sent():
    today = dt.date.today()
    current_year = (int(today.strftime("%Y")) + 1)
    start_year = today - dt.timedelta(days=365 * 20)
    start_year = (int(start_year.strftime("%Y")) + 1)

    # Get API key
    load_dotenv("nyt.env")
    api_key = os.getenv("NYT")
    url = "https://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}&q=economy"

    # Create an empty dataframe
    nyt_sentiment = pd.DataFrame(columns=["headline", "date", "compound"])

    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Loop through each year and each month from 2003 to today
    for year in range(start_year, current_year):
        for month in range(1, 13):
            # Format the year and month as a string
            year_month = f"{year}/{month:02d}"

            # Make a request to the NYT API
            response = requests.get(url.format(year, month, api_key))

            if response.status_code == 200:
                data = response.json()
                articles = data["response"]["docs"]

                # Loop through each article and add the headline, date, and sentiment scores to the nyt_sentiment dataframe
                for article in articles:
                    # Check if the article is about the economy
                    testers = ['economy', 's&p', 'stock market', 'us economy']
                    if any([x in article["snippet"].lower() for x in testers]) or any(
                            [x in article["abstract"].lower() for x in testers]):
                        # Get the sentiment scores for the headline and add them to the nyt_sentiment dataframe
                        scores = analyzer.polarity_scores(article["headline"]["main"] + " " + article["abstract"])
                        nyt_sentiment = nyt_sentiment.append({
                            "headline": article["headline"]["main"],
                            "date": pd.to_datetime(article["pub_date"]).date(),
                            "compound": scores["compound"]}, ignore_index=True)

    # Set date as index
    nyt_sentiment = nyt_sentiment.set_index('date')

    # Make a new dataframe called sentiment_df that groups the average sentiment for each day
    sentiment_df = nyt_sentiment.groupby('date').mean().reset_index()
    sentiment_df = sentiment_df.rename(columns={'compound': 'Sentiment'})

    return sentiment_df

