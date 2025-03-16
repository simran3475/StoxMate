import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from datetime import timedelta
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk  
import re


load_dotenv()
SERP_API_KEY = os.getenv("SERP_API_KEY")


nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()


st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")

# Sidebar elements
st.sidebar.title("Stock Market Predictor")
st.sidebar.write("Analyze stock prices, predict future trends using sentiment analysis, and include your own sentiments.")
stock_ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL, TSLA):')
st.sidebar.markdown("[Search Stock Tickers](https://finance.yahoo.com/lookup/)")
user_sentiment = st.sidebar.text_area("Enter your custom sentiment (e.g., 'The stock will rise by 5% in the next 3 months'):") 

@st.cache_data(ttl=3600)
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    try:
        info = stock.info
        return info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_stock_history(ticker):
    stock = yf.Ticker(ticker)
    try:
        stock_data = stock.history(period="6mo")
        return stock_data
    except Exception as e:
        st.error(f"Error fetching historical data for {ticker}: {e}")
        return None

def fetch_sentiments(stock_ticker):
    url = f"https://serpapi.com/search.json?engine=google_news&q={stock_ticker}&api_key={SERP_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        sentiments = [item['title'] for item in data.get('news_results', [])[:5]]
        return sentiments
    except Exception as e:
        st.error(f"Error fetching sentiments for {stock_ticker}: {e}")
        return []

def perform_sentiment_analysis(user_sentiment):
    sentiment_score = sia.polarity_scores(user_sentiment)['compound']
    return sentiment_score

def extract_percentage_change(user_sentiment):
    """
    Extracts percentage change from user sentiment.
    If no valid percentage is found, it defaults to a small change.
    Also considers specific keywords indicating a decline.
    """
  
    decline_keywords = ["fall", "drop", "decline", "decrease", "down", "losing"]
    increase_keywords = ["rise", "increase", "gain", "up", "gaining"]

    if any(keyword in user_sentiment.lower() for keyword in decline_keywords):
        match = re.search(r'([-+]?\d*\.?\d+)%?', user_sentiment)
        if match:
            return -float(match.group(0).replace('%', '')) / 100  
        return -0.05  

    if any(keyword in user_sentiment.lower() for keyword in increase_keywords):
        match = re.search(r'([-+]?\d*\.?\d+)%?', user_sentiment)
        if match:
            return float(match.group(0).replace('%', '')) / 100  
        return 0.05  

    return 0.05  

def predict_stock_price(stock_data, sentiment_factor):
    last_price = stock_data['Close'][-1]
    predicted_prices = []
    for i in range(1, 121):
        change = sentiment_factor * i / 120
        predicted_prices.append(last_price * (1 + change))
    future_dates = pd.date_range(stock_data.index[-1] + timedelta(days=1), periods=120)
    predicted_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Price'])
    return predicted_df


st.title('📊 Stock Market Predictor using Sentimental Analysis')
st.write("Analyze stock fundamentals, visualize historical data, and predict future trends based on market sentiments.")

if stock_ticker:
    stock_info = fetch_stock_data(stock_ticker)
    
    if stock_info:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📈 Stock Information for {stock_ticker}")
            st.markdown(f"""
            **Company Name:** {stock_info.get('longName', 'N/A')}
            
            **Sector:** {stock_info.get('sector', 'N/A')}
            
            **Market Cap:** {stock_info.get('marketCap', 'N/A')}
            
            **Current Price:** ${stock_info.get('previousClose', 'N/A'):,}
            """)

        with col2:
            st.subheader(f"📅 Historical Data (Last 6 Months)")
            stock_data = fetch_stock_history(stock_ticker)
            
            if stock_data is not None:
                st.line_chart(stock_data['Close'])
                st.write("The above chart shows the closing prices over the last 6 months.")
            else:
                st.write("Unable to retrieve historical stock data.")

        st.subheader(f"📰 Sentiment Analysis for {stock_ticker}")
        sentiments = fetch_sentiments(stock_ticker)
        
        if sentiments:
            for sentiment in sentiments:
                st.write(f"- {sentiment}")
        else:
            st.write("No sentiments found for this stock.")

        if user_sentiment:
            st.subheader("💬 User Custom Sentiment")
            st.write(f"Your Sentiment: {user_sentiment}")

            sentiment_factor = extract_percentage_change(user_sentiment)  

            st.subheader("📈 Predicted Stock Price for the Next 3-4 Months")
            if stock_data is not None:
                predicted_df = predict_stock_price(stock_data, sentiment_factor)
                combined_df = pd.concat([stock_data['Close'], predicted_df['Predicted Price']])
                st.line_chart(combined_df)
                
                st.subheader("📊 Table of Stock Prices")
                table_df = pd.concat([stock_data[['Close']], predicted_df.rename(columns={'Predicted Price': 'Predicted Price'})])
                st.write(table_df)
            else:
                st.write("Prediction not available due to missing historical data.")
    else:
        st.write("Invalid stock ticker or unable to fetch stock data.")

st.sidebar.markdown("Created by Satvik, Simran, and Yashi")