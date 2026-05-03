import streamlit as st
import pandas as pd
import numpy as np
import feedparser
from tradingview_screener import Query, col
import streamlit.components.v1 as components
import google.generativeai as genai
import plotly.graph_objects as go


st.set_page_config(page_title="SwingTrade Pro | Institutional Terminal", layout="wide")

#ai config
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
ai_model = genai.GenerativeModel('gemini-2.5-flash')
#custom CSS
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

#data scanner
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

#decision tree
def create_plotly_decision_tree():
    """
    Builds a fully interactive Plotly decision tree for swing trade entry logic.
    Layout uses a top-down flow with YES paths going down and NO/Caution
    branches going to the right.
    """

    # ── Node registry ──────────────────────────────────────────────────────────
    # Each entry: id, x, y, label (HTML-ish via <br>), color, text_color
    nodes = [
        # ── Row 0 — Start ──────────────────────────────────────────────────────
        dict(id="start",
             x=0.40, y=1.00,
             label="🔍 START<br>Review Tomorrow's Close",
             color="#2563eb", tc="white"),

        # ── Row 1 — Q1 ─────────────────────────────────────────────────────────
        dict(id="q1",
             x=0.40, y=0.85,
             label="1️⃣ Is Tomorrow's<br>Candle GREEN?",
             color="#1e40af", tc="white"),
        dict(id="q1_no",
             x=0.82, y=0.85,
             label="❌ AVOID<br>Wait Another Day",
             color="#dc2626", tc="white"),

        # ── Row 2 — Q2 ─────────────────────────────────────────────────────────
        dict(id="q2",
             x=0.40, y=0.70,
             label="2️⃣ Is RSI > 45<br>& Turning Up?",
             color="#1e40af", tc="white"),
        dict(id="q2_no",
             x=0.82, y=0.70,
             label="❌ AVOID<br>RSI Too Weak",
             color="#dc2626", tc="white"),

        # ── Row 3 — Q3 ─────────────────────────────────────────────────────────
        dict(id="q3",
             x=0.40, y=0.55,
             label="3️⃣ Is Price Above<br>MA Ribbon?",
             color="#1e40af", tc="white"),
        dict(id="q3_no",
             x=0.82, y=0.55,
             label="❌ AVOID<br>Below MA Support",
             color="#dc2626", tc="white"),

        # ── Row 4 — Q4 ─────────────────────────────────────────────────────────
        dict(id="q4",
             x=0.40, y=0.40,
             label="4️⃣ Is NASDAQ (NDQ)<br>Also Green?",
             color="#1e40af", tc="white"),
        dict(id="q4_no",
             x=0.82, y=0.40,
             label="⚠️ CAUTION<br>Market Headwind",
             color="#d97706", tc="white"),

        # ── Row 5 — Q5 ─────────────────────────────────────────────────────────
        dict(id="q5",
             x=0.40, y=0.25,
             label="5️⃣ Any Earnings<br>News Today?",
             color="#1e40af", tc="white"),
        dict(id="q5_yes",
             x=0.82, y=0.25,
             label="❌ AVOID<br>Earnings Risk",
             color="#dc2626", tc="white"),

        # ── Row 6 — Q6 ─────────────────────────────────────────────────────────
        dict(id="q6",
             x=0.40, y=0.10,
             label="6️⃣ Is Volume<br>> 2,000,000?",
             color="#1e40af", tc="white"),
        dict(id="q6_no",
             x=0.82, y=0.10,
             label="⚠️ CAUTION<br>Low Conviction",
             color="#d97706", tc="white"),

        # ── Final — BUY ────────────────────────────────────────────────────────
        dict(id="buy",
             x=0.40, y=-0.07,
             label="✅ PLACE BUY STOP<br>Entry: 1.002 × High<br>🔴 Stop: -5%  |  🎯 Target: +15%",
             color="#16a34a", tc="white"),
    ]

    # ── Edge registry ──────────────────────────────────────────────────────────
    # Each entry: from_id, to_id, label, label_color
    edges = [
        # Main YES flow (vertical)
        ("start", "q1",     "",         "white"),
        ("q1",    "q2",     "YES ✅",   "#4ade80"),
        ("q2",    "q3",     "YES ✅",   "#4ade80"),
        ("q3",    "q4",     "YES ✅",   "#4ade80"),
        ("q4",    "q5",     "YES ✅",   "#4ade80"),
        ("q5",    "q6",     "NO  ✅",   "#4ade80"),   # No earnings = good
        ("q6",    "buy",    "YES ✅",   "#4ade80"),

        # NO / Caution branches (horizontal)
        ("q1",    "q1_no",  "NO ❌",    "#f87171"),
        ("q2",    "q2_no",  "NO ❌",    "#f87171"),
        ("q3",    "q3_no",  "NO ❌",    "#f87171"),
        ("q4",    "q4_no",  "NO ⚠️",   "#fbbf24"),
        ("q5",    "q5_yes", "YES ❌",   "#f87171"),
        ("q6",    "q6_no",  "NO ⚠️",   "#fbbf24"),
    ]

    # ── Build lookup: id → (x, y) ──────────────────────────────────────────────
    pos = {n["id"]: (n["x"], n["y"]) for n in nodes}

    # ── Plotly figure ──────────────────────────────────────────────────────────
    fig = go.Figure()

    # Draw edges first (so nodes sit on top)
    for (src, dst, elabel, ecol) in edges:
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        mx = (x0 + x1) / 2          # midpoint for label
        my = (y0 + y1) / 2

        # Arrow line
        fig.add_shape(
            type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="rgba(255,255,255,0.6)", width=1.8),
            layer="below"
        )

        # Arrowhead annotation (invisible text, just the arrow)
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref="x", yref="y",
            axref="x", ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.4,
            arrowwidth=1.8,
            arrowcolor="rgba(255,255,255,0.7)"
        )

        # Edge label
        if elabel:
            # Offset label slightly so it doesn't sit on the line
            offset_x = 0.03 if x1 > x0 else -0.03
            offset_y = 0.015 if y1 == y0 else 0.01
            fig.add_annotation(
                x=mx + offset_x,
                y=my + offset_y,
                text=f"<b>{elabel}</b>",
                showarrow=False,
                font=dict(size=10, color=ecol),
                bgcolor="rgba(15,23,42,0.75)",
                borderpad=3,
                xref="x", yref="y"
            )

    # Draw node boxes via scatter (marker symbol = square, sized large)
    for n in nodes:
        x, y = n["x"], n["y"]

        # Shadow / glow effect
        fig.add_shape(
            type="rect",
            x0=x - 0.175, y0=y - 0.063,
            x1=x + 0.175, y1=y + 0.063,
            fillcolor="rgba(0,0,0,0.25)",
            line=dict(width=0),
            layer="below"
        )

        # Main box
        fig.add_shape(
            type="rect",
            x0=x - 0.170, y0=y - 0.058,
            x1=x + 0.170, y1=y + 0.058,
            fillcolor=n["color"],
            line=dict(color="white", width=1.5),
            layer="above"
        )

        # Node text
        fig.add_annotation(
            x=x, y=y,
            text=f"<b>{n['label']}</b>",
            showarrow=False,
            font=dict(size=11, color=n["tc"], family="Arial"),
            xref="x", yref="y",
            align="center"
        )

    # ── Legend boxes (bottom-left) ─────────────────────────────────────────────
    legend_items = [
        ("#16a34a", "✅ Buy Signal"),
        ("#dc2626", "❌ Avoid — Hard Stop"),
        ("#d97706", "⚠️ Caution — Use Discretion"),
        ("#1e40af", "🔵 Decision Checkpoint"),
        ("#2563eb", "🔍 Start Node"),
    ]
    for i, (lc, lt) in enumerate(legend_items):
        lx = -0.05
        ly = -0.18 - i * 0.055
        fig.add_shape(type="rect",
                      x0=lx, y0=ly - 0.018,
                      x1=lx + 0.04, y1=ly + 0.018,
                      fillcolor=lc,
                      line=dict(color="white", width=1))
        fig.add_annotation(x=lx + 0.055, y=ly,
                           text=f"<b>{lt}</b>",
                           showarrow=False,
                           font=dict(size=10, color="white"),
                           xref="x", yref="y",
                           xanchor="left")

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>📊 Swing Trade Entry Decision Tree</b><br>"
                 "<sup>Follow this checklist at tomorrow's market close</sup>",
            font=dict(size=18, color="white"),
            x=0.5, xanchor="center"
        ),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        xaxis=dict(
            visible=False,
            range=[-0.10, 1.10]
        ),
        yaxis=dict(
            visible=False,
            range=[-0.50, 1.10]
        ),
        height=900,
        margin=dict(l=20, r=20, t=80, b=20),
        showlegend=False
    )

    return fig


# --- 4. RISK CONTROL SIDEBAR ---
with st.sidebar:
    st.header("🛡️ Risk Management")
    capital = st.number_input("Account Balance ($)", value=10000, step=1000)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0, step=0.1)
    st.divider()
    st.info(f"Risk Amount: ${capital * (risk_pct/100):,.2f}")

#main table interface
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

    #chart and trade plan
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

    # --- ROW: DECISION TREE ---
    st.divider()
    st.subheader("🌳 Swing Trade Entry Decision Tree")
    st.caption("Follow this decision tree at tomorrow's market close before placing any trade.")

    tree_fig = create_plotly_decision_tree()
    st.plotly_chart(tree_fig, use_container_width=True)

    # How-to guide below the chart
    st.markdown("""
    <div class="checklist-container">
        <h3 style="margin-top:0; color:#1e293b; font-size:1.1rem;">📖 How To Use This Decision Tree</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; color:#1e293b; font-size:0.95rem;">
            <div>✅ <b>All 6 checks pass</b> → Place Buy Stop at 1.002 × High</div>
            <div>❌ <b>Any RED check fails</b> → Do not trade, wait another day</div>
            <div>⚠️ <b>CAUTION checks</b> → Reduce position size, use discretion</div>
            <div>📅 <b>When to check</b> → Review every day at market close (4PM ET)</div>
            <div>📊 <b>RSI Rule</b> → Must be above 45 AND visually turning upward</div>
            <div>🚫 <b>Earnings Rule</b> → Always avoid trading on earnings days</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Scanning for liquidity... no matches found.")
