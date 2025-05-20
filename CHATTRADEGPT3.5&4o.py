
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import requests
import pandas as pd
import datetime
from math import log, sqrt, exp
from bs4 import BeautifulSoup
import cloudscraper
import random
import time

# Fixed date context for analysis
FIXED_DATE = "Sunday, March 30, 2025"

# ----------------------------
# OpenAI API HTTP CALL FUNCTIONS
# ----------------------------
API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = "https://api.openai.com/v1/chat/completions"

# Available models
MODELS = {
    "gpt-3.5-turbo": "Standard analysis model",
    "gpt-40-mini": "Research model with web search capabilities",
    "gpt-4o": "Most advanced model for high-quality analysis"
}

def get_chatgpt_insight(prompt: str, max_tokens: int = 1500) -> str:
    """
    Sends a prompt to GPT-3.5-turbo via HTTP POST and returns the generated text.
    This function is used for main analysis.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"[GPT-3.5 HTTP Error]: {e}")
        return "AI analysis unavailable"

def get_research_insight(prompt: str, max_tokens: int = 1000) -> str:
    """
    Sends a prompt to GPT-40-mini via HTTP POST and returns the generated text.
    This function is used for research purposes where the model will directly look up web data.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-40-mini",  # Best model for research (web search)
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"[GPT-40-mini HTTP Error]: {e}")
        return "Research analysis unavailable"

def get_advanced_insight(prompt: str, max_tokens: int = 2000) -> str:
    """
    Sends a prompt to GPT-4o via HTTP POST and returns the generated text.
    This function is used for high-quality, advanced analysis when precision is critical.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o",  # Most advanced model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Lower temperature for more deterministic outputs
        "max_tokens": max_tokens
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=45)  # Longer timeout for GPT-4o
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"[GPT-4o HTTP Error]: {e}")
        print("Falling back to GPT-3.5-turbo...")
        # Fallback to GPT-3.5-turbo if GPT-4o fails
        return get_chatgpt_insight(prompt, max_tokens)

# ----------------------------
# DYNAMIC USER-AGENT FUNCTIONS
# ----------------------------
def generate_random_ua():
    generators = [
        lambda: f"Mozilla/5.0 (Linux; Android {random.randint(8,13)}; {random.choice(['Pixel 5', 'Galaxy S21', 'OnePlus 9', 'Pixel 4', 'Galaxy Note 20'])}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(80,142)}.{random.randint(0,99)}.{random.randint(3000,7000)}.{random.randint(0,150)} Mobile Safari/537.36",
        lambda: f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(80,142)}.{random.randint(0,99)}.{random.randint(3000,7000)}.{random.randint(0,150)} Safari/537.36",
        lambda: f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(80,142)}.{random.randint(0,99)}.{random.randint(3000,7000)}.{random.randint(0,150)} Safari/537.36",
        lambda: f"Mozilla/5.0 (iPhone; CPU iPhone OS {random.randint(12,16)}_{random.randint(0,5)} like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{random.randint(12,16)}.0 Mobile/15E148 Safari/604.1"
    ]
    return random.choice(generators)()

def get_headers(ua):
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/"
    }

def test_site(url, ua):
    headers = get_headers(ua)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code
    except Exception as e:
        return f"Error: {e}"

def find_successful_ua(target_urls):
    attempt = 0
    while True:
        attempt += 1
        ua = generate_random_ua()
        statuses = {}
        all_ok = True
        for site_name, url in target_urls.items():
            status = test_site(url, ua)
            statuses[site_name] = status
            if status != 200:
                all_ok = False
        if all_ok:
            print(f"\nSUCCESS on attempt #{attempt}: Found User-Agent:")
            print(ua)
            for site, status in statuses.items():
                print(f"{site} returned HTTP {status}")
            return ua
        else:
            status_summary = ', '.join(f"{k}: {v}" for k, v in statuses.items())
            print(f"Attempt #{attempt}: {status_summary}")
        time.sleep(0.5)

def get_dynamic_ua():
    target_urls = {
        "YahooNews": "https://finance.yahoo.com/quote/AAPL/news",
        "YahooOptions": "https://finance.yahoo.com/quote/AAPL/options",
        "MarketWatch": "https://www.marketwatch.com/investing/stock/aapl"  # Added MarketWatch
    }
    return find_successful_ua(target_urls)

# ----------------------------
# DATA RETRIEVAL FUNCTIONS (Using Dynamic UA)
# ----------------------------
def fetch_stock_data(symbol: str) -> (float, list):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=3mo&interval=1d"
    ua = get_dynamic_ua()
    headers = {"User-Agent": ua}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        result = data["chart"]["result"][0]
        meta = result["meta"]
        current_price = meta["regularMarketPrice"]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
        df = pd.DataFrame({
            "date": pd.to_datetime(timestamps, unit="s"),
            "close": closes
        })
        df.set_index("date", inplace=True)
        last_10 = df["close"].tail(10).round(2).tolist()
        return current_price, last_10
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        sys.exit(1)

def fetch_real_time_news(symbol: str) -> list:
    url = f"https://finance.yahoo.com/quote/{symbol}/news"
    ua = get_dynamic_ua()
    headers = get_headers(ua)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        headlines = [h3.get_text().strip() for h3 in soup.find_all("h3") if h3.get_text().strip()]
        return headlines[:5]
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return []

def fetch_options_data(symbol: str) -> pd.DataFrame:
    url = f"https://finance.yahoo.com/quote/{symbol}/options"
    ua = get_dynamic_ua()
    headers = get_headers(ua)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            print("No options tables found.")
            return pd.DataFrame()
        data = []
        for table in tables:
            rows = table.find_all("tr")[1:]
            for row in rows:
                cols = [col.get_text().strip() for col in row.find_all("td")]
                if len(cols) >= 11:
                    data.append(cols)
        if not data:
            return pd.DataFrame()
        columns = ["Contract Name", "Last Trade Date", "Strike Price", "Last Price",
                   "Bid", "Ask", "Change", "% Change", "Volume", "Open Interest", "Implied Volatility"]
        df = pd.DataFrame(data, columns=columns)
        df["Type"] = df["Contract Name"].apply(lambda x: "Call" if "C" in x else ("Put" if "P" in x else "Unknown"))
        for col in ["Strike Price", "Last Price", "Bid", "Ask", "Volume", "Open Interest", "Implied Volatility"]:
            df[col] = pd.to_numeric(df[col].replace({'%':''}, regex=True), errors='coerce')
        return df
    except Exception as e:
        print(f"Error fetching options data for {symbol}: {e}")
        return pd.DataFrame()

# ----------------------------
# MARKETWATCH DATA RETRIEVAL FUNCTION
# ----------------------------
def fetch_marketwatch_data(symbol: str) -> dict:
    """
    Fetches stock data and news from MarketWatch for the given symbol.
    
    Args:
        symbol (str): The stock symbol to fetch data for.
        
    Returns:
        dict: A dictionary containing MarketWatch data.
    """
    url = f"https://www.marketwatch.com/investing/stock/{symbol.lower()}"
    ua = get_dynamic_ua()
    headers = get_headers(ua)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract data from MarketWatch
        data = {}
        
        # Get the current price
        price_elem = soup.select_one('.intraday__price .value')
        if price_elem:
            data['current_price'] = float(price_elem.get_text().replace(',', ''))
        
        # Get news headlines
        news_items = soup.select('.article__content')
        data['news'] = []
        for item in news_items[:5]:  # Get top 5 news
            headline = item.select_one('.article__headline')
            if headline:
                data['news'].append(headline.get_text().strip())
        
        # Get analyst ratings if available
        ratings = soup.select('.analyst__option')
        if ratings:
            data['analyst_ratings'] = {}
            for rating in ratings:
                rating_type = rating.select_one('.label')
                rating_count = rating.select_one('.data__value')
                if rating_type and rating_count:
                    data['analyst_ratings'][rating_type.get_text().strip()] = rating_count.get_text().strip()
        
        return data
    except Exception as e:
        print(f"Error fetching MarketWatch data for {symbol}: {e}")
        return {}

# ----------------------------
# MAIN ANALYSIS FUNCTION (GPT-3.5)
# ----------------------------
def main_analysis(symbol: str, budget: float, current_price: float, last_closes: list, news: list, marketwatch_data: dict = None) -> str:
    marketwatch_section = ""
    if marketwatch_data:
        marketwatch_section = "MarketWatch Data:\n"
        if 'current_price' in marketwatch_data:
            marketwatch_section += f"- MarketWatch price: ${marketwatch_data['current_price']:.2f}\n"
        if 'news' in marketwatch_data and marketwatch_data['news']:
            marketwatch_section += "- MarketWatch News Headlines:\n"
            for headline in marketwatch_data['news']:
                marketwatch_section += f"  * {headline}\n"
        if 'analyst_ratings' in marketwatch_data:
            marketwatch_section += "- Analyst Ratings:\n"
            for rating_type, count in marketwatch_data['analyst_ratings'].items():
                marketwatch_section += f"  * {rating_type}: {count}\n"

    prompt = f"""
Today is {FIXED_DATE}. You are an advanced financial analyst with full research capabilities.
For the stock symbol {symbol}, the current price is ${current_price:.2f} and the last 10 daily closing prices are:
{last_closes}.
Real-time news headlines for {symbol} are:
{news}.
{marketwatch_section}
The trading budget is ${budget:.2f}.

Please perform the following tasks:
1. Calculate technical indicators:
   - Annualized volatility (using daily log returns),
   - 20-day Simple Moving Average (SMA),
   - 14-day Relative Strength Index (RSI).
2. Using the current price as the strike, compute the Black–Scholes price for a European call option with 7 days until expiration.
   Assume a risk-free rate of 1% and use the calculated annualized volatility as the implied volatility.
3. Calculate the option Greeks for this call option: delta, gamma, vega (per 1% change), theta (per day), and rho.
4. Based on your analysis and current market trends from the news, recommend winning strategies for:
   - Call options,
   - Put options, and
   - Debit spreads.
   For each recommendation, provide:
      - Option or spread details (e.g., strike price, bid/ask if available),
      - A "chance of winning" metric (percentage),
      - A detailed rationale.
5. Provide a comprehensive summary report that includes all calculations, recommendations, and your assumptions.
Return your answer as a JSON object with the following keys:
"technical_indicators": {{
    "annual_volatility": <value>,
    "SMA_20": <value>,
    "RSI_14": <value>
}},
"option_pricing": {{
    "call_price": <value>
}},
"greeks": {{
    "delta": <value>,
    "gamma": <value>,
    "vega": <value>,
    "theta": <value>,
    "rho": <value>
}},
"recommendations": {{
    "calls": [{{"option_details": <string>, "chance_of_winning": <value>, "rationale": <string>}}],
    "puts": [{{"option_details": <string>, "chance_of_winning": <value>, "rationale": <string>}}],
    "debit_spreads": [{{"spread_details": <string>, "chance_of_winning": <value>, "rationale": <string>}}]
}},
"report_summary": "<detailed summary report text>"
Ensure all numeric values are rounded to two decimal places.
    """.strip()
    return get_chatgpt_insight(prompt, max_tokens=2000)

# ----------------------------
# RESEARCH FUNCTION (Using GPT-40-mini for Web Search)
# ----------------------------
def research_for_cheapest_option(symbol: str, options_df: pd.DataFrame, news: list, marketwatch_data: dict = None) -> str:
    marketwatch_section = ""
    if marketwatch_data:
        marketwatch_section = "MarketWatch Data:\n"
        if 'current_price' in marketwatch_data:
            marketwatch_section += f"- MarketWatch price: ${marketwatch_data['current_price']:.2f}\n"
        if 'news' in marketwatch_data and marketwatch_data['news']:
            marketwatch_section += "- MarketWatch News Headlines:\n"
            for headline in marketwatch_data['news']:
                marketwatch_section += f"  * {headline}\n"
        if 'analyst_ratings' in marketwatch_data:
            marketwatch_section += "- Analyst Ratings:\n"
            for rating_type, count in marketwatch_data['analyst_ratings'].items():
                marketwatch_section += f"  * {rating_type}: {count}\n"
    
    if options_df.empty:
        options_json = "No options data available."
    else:
        sample_df = options_df.head(20).to_dict(orient="records")
        options_json = json.dumps(sample_df)
    
    prompt = f"""
Today is {FIXED_DATE}. You are an expert market researcher with full web-search capabilities using GPT-40-mini.
Search the internet for the latest sentiment, price, and related information for {symbol}.
Then, analyze the following options chain data (in JSON format):
{options_json}
{marketwatch_section}
and consider these real-time news headlines:
{news}
Based on your research, identify the cheapest option contract (either Call or Put) that shows the highest chance of winning.
Return a JSON object containing:
  "option_type",
  "strike_price",
  "bid",
  "ask",
  "chance_of_winning",
  "rationale"
Round all numeric values to two decimals.
    """.strip()
    return get_research_insight(prompt, max_tokens=1000)

# ----------------------------
# INTERACTIVE CHAT MODE
# ----------------------------
def interactive_chat():
    print("\nInteractive Chat Mode: You may now ask follow-up questions about the analysis. Type 'exit' to quit.")
    print("Available models: 1) GPT-3.5-turbo (default), 2) GPT-4o (advanced), 3) GPT-40-mini (research)")
    
    while True:
        user_query = input("Your question: ").strip()
        
        if user_query.lower() == "exit":
            break
            
        # Check if user wants to use a specific model
        if user_query.startswith("!model"):
            parts = user_query.split(" ", 2)
            if len(parts) >= 3:
                model_choice = parts[1].strip()
                actual_query = parts[2].strip()
                
                if model_choice == "1":
                    print("Using GPT-3.5-turbo...")
                    answer = get_chatgpt_insight(actual_query, max_tokens=500)
                elif model_choice == "2":
                    print("Using GPT-4o (advanced model)...")
                    answer = get_advanced_insight(actual_query, max_tokens=1000)
                elif model_choice == "3":
                    print("Using GPT-40-mini (research model)...")
                    answer = get_research_insight(actual_query, max_tokens=800)
                else:
                    print("Invalid model choice. Using default GPT-3.5-turbo...")
                    answer = get_chatgpt_insight(actual_query, max_tokens=500)
            else:
                print("Invalid format. Use '!model [1/2/3] [your question]'")
                continue
        else:
            # Default to GPT-3.5-turbo
            answer = get_chatgpt_insight(user_query, max_tokens=500)
            
        print("\nResponse:")
        print(answer)
        print("-" * 50)

# ----------------------------
# USER INPUT FUNCTION
# ----------------------------
def get_user_input() -> (str, float):
    """Prompt the user for the stock symbol and trading budget."""
    try:
        symbol = input("Enter the stock symbol to analyze: ").strip().upper()
        if not symbol:
            raise ValueError("Stock symbol cannot be empty.")
        budget_str = input("Enter your trading budget ($): ").strip()
        budget = float(budget_str)
        if budget <= 0:
            raise ValueError("Budget must be a positive number.")
        return symbol, budget
    except ValueError as e:
        print(f"Input Error: {e}")
        sys.exit(1)

# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main():
    # 1. Get user input
    symbol, budget = get_user_input()

    # 2. Retrieve stock data
    current_price, last_closes = fetch_stock_data(symbol)
    print(f"\nCurrent price for {symbol}: ${current_price:.2f}")
    print(f"Last 10 closing prices: {last_closes}")

    # 3. Retrieve real-time news headlines
    news_headlines = fetch_real_time_news(symbol)
    if news_headlines:
        print("\nReal-Time News Headlines:")
        for headline in news_headlines:
            print(f"- {headline}")
    else:
        print("\nNo real-time news headlines retrieved.")

    # 4. Retrieve MarketWatch data
    marketwatch_data = fetch_marketwatch_data(symbol)
    if marketwatch_data:
        print("\nMarketWatch Data:")
        if 'current_price' in marketwatch_data:
            print(f"- MarketWatch price: ${marketwatch_data['current_price']:.2f}")
        if 'news' in marketwatch_data and marketwatch_data['news']:
            print("- MarketWatch News Headlines:")
            for headline in marketwatch_data['news']:
                print(f"  * {headline}")
        if 'analyst_ratings' in marketwatch_data:
            print("- Analyst Ratings:")
            for rating_type, count in marketwatch_data['analyst_ratings'].items():
                print(f"  * {rating_type}: {count}")
    else:
        print("\nNo MarketWatch data retrieved.")

    # 5. Retrieve options chain data
    options_df = fetch_options_data(symbol)
    if options_df.empty:
        print("No options data retrieved from the marketplace.")
    else:
        print(f"\nRetrieved {len(options_df)} options records from the marketplace.")

    # 6. Get main analysis report from GPT-3.5
    print("\nGenerating main analysis with GPT-3.5-turbo...")
    analysis_report = main_analysis(symbol, budget, current_price, last_closes, news_headlines, marketwatch_data)

    # 7. Get research on the cheapest winning option using GPT-40-mini
    print("\nResearching cheapest winning option with GPT-40-mini...")
    cheapest_option_report = research_for_cheapest_option(symbol, options_df, news_headlines, marketwatch_data)
    
    # 8. Get advanced options strategy using GPT-4o
    print("\nGenerating advanced options strategy with GPT-4o...")
    strategy_prompt = f"""
Today is {FIXED_DATE}. You are an elite options trading strategist with deep expertise.
For stock symbol {symbol} (current price: ${current_price:.2f}), analyze this data:

Last 10 closing prices: {last_closes}
News: {news_headlines}
Budget: ${budget:.2f}

Based on this information and your expertise:
1. Identify the optimal advanced options strategy for current market conditions
2. Explain specific entry/exit points and exact option contracts to use
3. Calculate expected profit/loss scenarios with probabilities
4. Provide risk management rules specific to this trade

Return your analysis as a JSON object with these keys:
"strategy_name", "contracts", "entry_points", "exit_points", "profit_potential", "loss_risk", 
"probability_of_success", "key_risks", "detailed_rationale"
    """.strip()
    
    try:
        advanced_strategy = get_advanced_insight(strategy_prompt)
    except Exception as e:
        print(f"Error generating advanced strategy: {e}")
        advanced_strategy = "Advanced strategy analysis unavailable."

    # 9. Display analysis report
    try:
        analysis_json = json.loads(analysis_report)
        print("\n===== Trading Analysis Report (GPT-3.5) =====")
        print(json.dumps(analysis_json, indent=2))
    except Exception:
        print("\n===== Trading Analysis Report (Raw Text) =====")
        print(analysis_report)

    # 10. Display cheapest option research report
    try:
        cheapest_json = json.loads(cheapest_option_report)
        print("\n===== Cheapest Option Research Report (GPT-40-mini) =====")
        print(json.dumps(cheapest_json, indent=2))
    except Exception:
        print("\n===== Cheapest Option Research Report (Raw Text) =====")
        print(cheapest_option_report)
    
    # 11. Display advanced strategy from GPT-4o
    try:
        advanced_json = json.loads(advanced_strategy)
        print("\n===== Advanced Options Strategy (GPT-4o) =====")
        print(json.dumps(advanced_json, indent=2))
    except Exception:
        print("\n===== Advanced Options Strategy (Raw Text) =====")
        print(advanced_strategy)

    # 12. Enter interactive chat mode for follow-up questions
    interactive_chat()

if __name__ == "__main__":
    main()