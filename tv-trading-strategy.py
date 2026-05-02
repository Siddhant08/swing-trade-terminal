import streamlit as st
import pandas as pd
import numpy as np
import feedparser
from tradingview_screener import Query, col
import streamlit.components.v1 as components
import google.generativeai as genai

# --- 1. ARCHITECTURE & AI SETUP ---
st.set_page_config(page_title="SwingTrade Pro | Institutional Terminal", layout="wide")

# API Configuration
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
ai_model = genai.GenerativeModel('gemini-2.5-flash')

# Custom CSS for Professional Terminal & Table Styling
st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    
    /* Clean Table Styling */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        border: 2px solid black;
        font-family: sans-serif;
    }
    .custom-table thead th {
        background-color: #008c39;
        color: white;
        font-weight: bold;
        padding: 12px;
        border: 2px solid black;
        text-align: left;
    }
    .custom-table tbody td {
        padding: 10px;
        border: 2px solid black;
        background-color: white;
        color: #1e293b;
    }
    .custom-table tbody tr:hover {
        background-color: #f1f5f9;
    }

    /* Container Card Styling */
    .plan-card { 
        background-color: #ffffff; padding: 24px; border-radius: 12px; 
        border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Checklist Styling */
    .checklist-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #2563eb;
        margin-top: 10px;
    }

    .ai-insight-box {
        background-color: #f1f5f9; padding: 15px; border-radius: 8px;
        border-left: 4px solid #0f172a; font-size: 0.95rem; margin-top: 15px;
    }
    .news-link { color: #2563eb; text-decoration: none; font-weight: 500; display: block; margin-bottom: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINES ---
@st.cache_data(ttl=3600)
def get_market_data():
    try:
        search = (
            Query()
            .select('name', 'description', 'close', 'high', 'low', 'sector', 'change', 'volume', 'RSI')
            .where(col('close') > 5.0, col('volume') > 2000000, col('RSI').between(40, 80))
            .order_by('volume', ascending=False).limit(20)
        )
        return search.get_scanner_data()[1].dropna()
    except Exception:
        return pd.DataFrame()

def get_ticker_news(symbol):
    feed_url = f"https://news.google.com/rss/search?q={symbol}+stock+news+when:7d&hl=en-US&gl=US&ceid=US:en"
    return feedparser.parse(feed_url).entries[:3]

def get_ai_analysis(name, ticker, close, rsi):
    prompt = f"""Act as a Senior Quant Analyst. Provide a concise swing trade verdict for {name} ({ticker}).
    Current Price: ${close}, RSI: {rsi:.1f}.
    Analyze based on 2026 market sentiment, institutional flow, and technicals.
    Format: 1 sentence 'Verdict' (Buy/Hold/Avoid) followed by 3 short bullet points.
    Be concise, objective, and professional."""
    try:
        response = ai_model.generate_content(prompt)
        return response.text
    except Exception:
        return "Analyst unavailable."

# --- 3. RISK CONTROL SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Risk Management")
    capital = st.number_input("Account Balance ($)", value=10000, step=1000)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0, step=0.1)
    st.divider()
    st.info(f"Risk Amount: ${capital * (risk_pct/100):,.2f}")

# --- 4. MAIN ANALYTICS INTERFACE ---
st.title("📈 Institutional Swing Terminal")
df_raw = get_market_data()

if not df_raw.empty:
    df = df_raw.copy()
    df['Entry'] = (df['high'] * 1.002).round(2)
    df['Stop'] = (df['Entry'] * 0.95).round(2)
    df['Target'] = (df['Entry'] * 1.15).round(2)
    df['Shares'] = ((capital * (risk_pct/100)) / (df['Entry'] - df['Stop'] + 0.001)).astype(int)

    st.subheader("📊 Market Scan: High-Volume Momentum")
    
    headers = ["Ticker", "Sector", "Price", "Chg %", "Buy Stop", "Qty", "RSI (14)"]
    
    table_html = '<table class="custom-table"><thead><tr>'
    for h in headers:
        table_html += f'<th>{h}</th>'
    table_html += '</tr></thead><tbody>'
    
    for _, row in df.iterrows():
        table_html += '<tr>'
        table_html += f'<td>{row["name"]}</td>'
        table_html += f'<td>{row["sector"]}</td>'
        table_html += f'<td>${row["close"]:.2f}</td>'
        table_html += f'<td>{row["change"]:.2f}%</td>'
        table_html += f'<td>${row["Entry"]:.2f}</td>'
        table_html += f'<td>{int(row["Shares"])}</td>'
        table_html += f'<td>{row["RSI"]:.1f}</td>'
        table_html += '</tr>'
    
    table_html += '</tbody></table>'
    st.markdown(table_html, unsafe_allow_html=True)

    st.divider()

    # --- ROW: CHART & TRADE PLAN ---
    col_chart, col_plan = st.columns([2, 1])

    with col_plan:
        st.subheader("🎯 Trade Plan")
        selected = st.selectbox("Active Ticker:", df['name'].tolist())
        row_sel = df[df['name'] == selected].iloc[0]
        
        st.markdown(f"""
        <div class="plan-card">
            <h2 style="margin:0; color:#1e293b;">{selected}</h2>
            <p style="color:#64748b; font-size:0.85rem; margin-bottom:15px;">{row_sel['description']}</p>
            <hr style="border:0; border-top:1px solid #e2e8f0; margin:15px 0;">
            <p><b>🟢 Entry:</b> ${row_sel['Entry']:.2f}</p>
            <p><b>🔴 Stop Loss:</b> ${row_sel['Stop']:.2f}</p>
            <p><b>🎯 Target:</b> ${row_sel['Target']:.2f}</p>
            <p><b>📦 Position:</b> {int(row_sel['Shares'])} Shares</p>
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        st.subheader(f"🔍 {selected} Technical Analysis")
        html_code = f"""
        <div id="tv_chart"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({{
          "width": "100%", "height": 400, "symbol": "{selected}",
          "interval": "D", "theme": "light", "style": "1",
          "container_id": "tv_chart",
          "studies": ["Moving Average Ribbon@tv-basicstudies"]
        }});
        </script>
        """
        components.html(html_code, height=410)
        
        # --- RESTORED CHECKLIST ---
        st.markdown(f"""
        <div class="checklist-container">
            <h3 style="margin-top:0; color:#1e293b; font-size:1.1rem;">🚀 Pre-Flight Checklist: {selected}</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="display:flex; align-items:center; gap:8px;"><input type="checkbox"> <span>Price above MA Ribbon</span></div>
                <div style="display:flex; align-items:center; gap:8px;"><input type="checkbox"> <span>RSI not Overbought (>70)</span></div>
                <div style="display:flex; align-items:center; gap:8px;"><input type="checkbox"> <span>Institutional Volume Support</span></div>
                <div style="display:flex; align-items:center; gap:8px;"><input type="checkbox"> <span>No major earnings news today</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- ROW: NEWS & AI BRIEFING ---
    st.divider()
    col_news, col_ai = st.columns([1, 1])

    with col_news:
        st.subheader(f"🗞️ {selected} News Brief")
        news_items = get_ticker_news(selected)
        if news_items:
            for item in news_items:
                st.markdown(f'<a class="news-link" href="{item.link}" target="_blank">🔗 {item.title}</a>', unsafe_allow_html=True)
        else:
            st.info("No recent news found.")

    with col_ai:
        st.subheader("🤖 GenAI Analyst Insight")
        if st.button("✨ Execute AI Deep-Dive", use_container_width=True):
            with st.spinner("Analyzing 2026 Market Data..."):
                analysis = get_ai_analysis(row_sel['description'], selected, row_sel['close'], row_sel['RSI'])
                st.markdown(f"""<div class="ai-insight-box"><b>Quant Intelligence Brief:</b><br>{analysis}</div>""", unsafe_allow_html=True)
else:
    st.warning("Scanning for liquidity... no matches found.")
