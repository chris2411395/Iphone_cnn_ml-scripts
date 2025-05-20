# Iphone_cnn_ml-scripts
Custom iPhone scripts for CNN/ML implementation with licensing tiers
# Advanced Trading AI Client

This Python script is your all-in-one trading analysis toolkit. It combines:

- **Live data scraping** (stock prices, news, options chains)  
- **Dynamic user-agent rotation** for reliable web requests  
- **Automated technical analysis** (volatility, SMA, RSI, Black–Scholes, Greeks)  
- **Multi-model AI insights** via OpenAI’s GPT endpoints  
- **Interactive chat mode** for follow-up Q&A

—

## 🚀 Features Overview

1. **OpenAI Integration**  
   - `get_chatgpt_insight()` (GPT-3.5-turbo) for primary analysis  
   - `get_research_insight()` (GPT-40-mini) for web-enabled research  
   - `get_advanced_insight()` (GPT-4o) for high-precision strategic advice  

2. **Dynamic User-Agent Rotation**  
   - `generate_random_ua()` picks random Chrome/Safari UA strings  
   - `find_successful_ua()` probes Yahoo, MarketWatch, etc., until a 200 OK is found  

3. **Market Data Fetchers**  
   - `fetch_stock_data(symbol)`  
     > Pulls 3-month daily closes and current price from Yahoo Finance JSON API  
   - `fetch_real_time_news(symbol)`  
     > Scrapes top 5 headlines from Yahoo Finance  
   - `fetch_options_data(symbol)`  
     > Parses calls/puts tables from Yahoo Finance options page into a Pandas DataFrame  
   - `fetch_marketwatch_data(symbol)`  
     > Scrapes current price, headlines, and analyst ratings from MarketWatch  

4. **Core Analysis Pipeline**  
   - `main_analysis(...)`  
     - Calculates annualized volatility, 20-day SMA, 14-day RSI  
     - Prices a 7-day European call (Black–Scholes)  
     - Computes Greeks: delta, gamma, vega, theta, rho  
     - Asks GPT-3.5 to produce JSON-formatted recommendations (calls, puts, debit spreads)  
   - `research_for_cheapest_option(...)`  
     - Uses GPT-40-mini to research live web sentiment and find the cheapest “high-win-rate” option  

5. **Interactive Follow-Up Chat**  
   - `interactive_chat()` lets you query the AI models directly (choose between GPT-3.5, GPT-4o, GPT-40-mini)  

6. **User Input & CLI**  
   - Prompts for stock symbol & budget  
   - Prints formatted JSON reports for analysis, research, and advanced strategy  

—

## ⚙️ Requirements

- Python 3.10+  
- Environment variable:  
  ```bash
  export OPENAI_API_KEY=“sk-...”
  
  requests
pandas
beautifulsoup4
cloudscraper
git clone https://github.com/chris2411395/iphone_cnn_ml-scripts.git
cd iphone_cnn_ml-scripts
pip install -r requirements.txt
1.	Enter the stock symbol when prompted (e.g., AAPL).
	2.	Enter your trading budget (e.g., 1000).
	3.	The script will:
	•	Fetch price, closes, news, options, MarketWatch info
	•	Call GPT-3.5 for the main analysis
	•	Call GPT-40-mini for cheapest option research
	•	Call GPT-4o for advanced options strategy
	•	Display all reports as JSON in your console
	4.	Type follow-up questions in the interactive chat, or exit to quit.
	.
├── your_script_name.py    # This all-in-one analysis client
├── requirements.txt       # pip dependencies
└── README.md              # This documentation

  This code is provided as-is for educational/demo purposes.
Commercial use, redistribution, or modification of the AI logic requires a paid subscription.
  
  
  
  