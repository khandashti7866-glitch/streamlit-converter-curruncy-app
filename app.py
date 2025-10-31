# app.py
"""
Streamlit Global Currency Converter with Visual Analytics
Uses exchangerate.host (free, no API key) for live & historical rates.
Author: Generated for user
Run: streamlit run app.py
"""

from datetime import datetime, timedelta
import re
import io

import requests
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

# ---------------------------
# App config & style
# ---------------------------
st.set_page_config(
    page_title="Global Currency Converter — Visual Dashboard",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ultra-clean header styling (Streamlit markdown + inline CSS)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #07103a 50%, #020617 100%);
        color: #e6eef8;
    }
    .card {
        background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.7);
    }
    .metric {
        font-weight: 700;
        font-size: 20px;
        color: #fff;
    }
    .small {
        color: #b8c7e0;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='color:#eaf2ff'>💱 Global Currency Converter — Visual Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='small'>Live conversion, historical trends, top-10 comparisons, CSV export, and NLP input — all without an API key.</div>", unsafe_allow_html=True)
st.write("")

# ---------------------------
# Helper data & utilities
# ---------------------------

# A helpful list of currencies (common + many others). We'll fetch full list from API too.
@st.cache_data(ttl=24*3600)
def get_supported_symbols():
    url = "https://api.exchangerate.host/symbols"
    r = requests.get(url, timeout=10)
    data = r.json()
    symbols = data.get("symbols", {})
    # symbols: { 'USD': {'description': 'United States Dollar', 'code': 'USD'}, ...}
    return symbols

SYMBOLS = get_supported_symbols()

# Small currency symbol mapping (partial; fallback to currency code)
CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥", "INR": "₹", "AUD": "A$", "CAD": "C$", "CHF": "Fr",
    "PKR": "₨", "SGD": "S$", "HKD": "HK$", "NZD": "NZ$", "SEK": "kr", "KRW": "₩", "ZAR": "R"
}

# Basic currency -> approximate country code mapping for flag emoji (best-effort)
CURRENCY_TO_COUNTRY = {
    "USD": "US","EUR": "EU","GBP":"GB","JPY":"JP","CNY":"CN","INR":"IN","AUD":"AU","CAD":"CA","CHF":"CH",
    "PKR":"PK","SGD":"SG","HKD":"HK","NZD":"NZ","SEK":"SE","KRW":"KR","ZAR":"ZA","BRL":"BR","RUB":"RU",
    "MXN":"MX","IDR":"ID","TRY":"TR","ILS":"IL","SAR":"SA","AED":"AE","NGN":"NG","EGP":"EG"
}

def flag_emoji_for_currency(ccy):
    """Return a flag emoji for a currency code where possible."""
    code = CURRENCY_TO_COUNTRY.get(ccy)
    if not code:
        return ""
    # EU has no single flag in emoji standards, use a globe as fallback
    if code == "EU":
        return "🌍"
    # Convert country code to regional indicator symbols
    try:
        emoji = ''.join(chr(ord(c) + 127397) for c in code.upper())
        return emoji
    except Exception:
        return ""

def currency_symbol(ccy):
    return CURRENCY_SYMBOLS.get(ccy, ccy)

# ---------------------------
# API functions (cached)
# ---------------------------
BASE_API = "https://api.exchangerate.host"

@st.cache_data(ttl=60)  # cache live rates for 60 seconds by default
def fetch_latest_rates(base: str = "USD"):
    """Fetch latest rates for all currencies relative to base."""
    url = f"{BASE_API}/latest"
    params = {"base": base}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60*60)  # cache timeseries for 1 hour
def fetch_timeseries(base: str, target: str, days: int = 7):
    """Fetch historical timeseries for last `days` days (including today)."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=days-1)
    url = f"{BASE_API}/timeseries"
    params = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "base": base,
        "symbols": target
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# ---------------------------
# NLP: parse natural language like "convert 100 USD to EUR"
# ---------------------------
AMOUNT_CURRENCY_PATTERN = re.compile(
    r'(?P<amount>\d+(?:[\.,]\d+)?)\s*(?P<from>[A-Za-z]{3})\s*(?:to|-|=>|in)?\s*(?P<to>[A-Za-z]{3})?',
    flags=re.IGNORECASE
)

def parse_nl_input(text: str):
    """Try to parse an NL description. Returns (amount, from, to) or None values."""
    text = text.strip()
    m = AMOUNT_CURRENCY_PATTERN.search(text)
    if not m:
        return None, None, None
    amt = m.group("amount").replace(",", "")
    try:
        amount = float(amt)
    except:
        amount = None
    frm = (m.group("from") or "").upper()
    to = (m.group("to") or "").upper() if m.group("to") else None
    # Validate against SYMBOLS
    if frm not in SYMBOLS:
        # maybe user wrote currency name; try match by description
        for code, meta in SYMBOLS.items():
            if meta["description"].lower().startswith(frm.lower()):
                frm = code
                break
    if to and to not in SYMBOLS:
        for code, meta in SYMBOLS.items():
            if meta["description"].lower().startswith(to.lower()):
                to = code
                break
    return amount, frm, to

# ---------------------------
# Conversion logic
# ---------------------------
def convert_amount(amount: float, base: str, target: str, latest_json=None):
    """Convert amount using latest_json (if provided) else fetch rates."""
    if latest_json is None:
        latest_json = fetch_latest_rates(base)
    rates = latest_json.get("rates", {})
    if target not in rates:
        raise ValueError(f"Target currency {target} not available for base {base}.")
    rate = rates[target]
    converted = amount * rate
    return converted, rate

# ---------------------------
# UI: Sidebar controls
# ---------------------------
with st.sidebar:
    st.markdown("<div class='card'><h3>Controls</h3>", unsafe_allow_html=True)
    # Auto-detect currency? (not doing geolocation here)
    amount_input = st.text_input("Amount or description (e.g. '100', 'Convert 100 USD to EUR')", value="100")
    col1, col2 = st.columns(2)
    with col1:
        base_ccy = st.selectbox("Base currency", options=sorted(SYMBOLS.keys()), index=sorted(SYMBOLS.keys()).index("USD") if "USD" in SYMBOLS else 0)
    with col2:
        target_ccy = st.selectbox("Target currency", options=sorted(SYMBOLS.keys()), index=sorted(SYMBOLS.keys()).index("EUR") if "EUR" in SYMBOLS else 0)

    days = st.radio("Historical trend days", options=[7, 14, 30], index=0, horizontal=True)
    refresh_manual = st.button("Refresh Rates Now 🔄")
    auto_refresh = st.checkbox("Auto-refresh every 5 minutes (keeps rates up-to-date)", value=False)
    st.markdown("</div>", unsafe_allow_html=True)

# If user entered NL phrase, attempt parse
nl_amount, nl_from, nl_to = parse_nl_input(amount_input)

# If parsed and complete, override fields
if nl_amount and nl_from and nl_to:
    # use parsed values
    try:
        base_ccy = nl_from
        target_ccy = nl_to
        amount = nl_amount
    except Exception:
        amount = float(re.sub("[^0-9\.]", "", amount_input)) if re.search(r'\d', amount_input) else 1.0
else:
    # if amount_input is just a number or decimal, parse it
    num_match = re.search(r'\d+(?:[\.,]\d+)?', amount_input)
    amount = float(num_match.group(0).replace(",", "")) if num_match else 1.0

# ---------------------------
# Fetch latest rates (with manual refresh option)
# ---------------------------
# If user clicked refresh, clear cache for fetch_latest_rates and fetch fresh
if refresh_manual:
    # Simple way: call function with different argument to bypass cache:
    fetch_latest_rates.clear()  # clear cached data
    latest_data = fetch_latest_rates(base_ccy)
else:
    latest_data = fetch_latest_rates(base_ccy)

# If auto_refresh is checked, we still rely on cache ttl; let user know
if auto_refresh:
    st.sidebar.info("Auto-refresh uses cached endpoints (cached for short intervals).")

# ---------------------------
# Main dashboard
# ---------------------------
# Top metrics container
top1, top2 = st.columns([3, 2])
with top1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # Conversion card
    try:
        converted_value, used_rate = convert_amount(amount, base_ccy, target_ccy, latest_data)
        symbol = currency_symbol(target_ccy)
        flag = flag_emoji_for_currency(target_ccy)
        st.markdown(f"<div style='display:flex;align-items:center;gap:12px'><div><h2 style='margin:0'>{flag} {symbol} {converted_value:,.4f}</h2></div><div class='small'>Converted from <strong>{amount:,.4f} {base_ccy}</strong> at rate <strong>1 {base_ccy} = {used_rate:.6f} {target_ccy}</strong></div></div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Conversion failed: {e}")
        st.stop()
    st.markdown("</div>", unsafe_allow_html=True)

with top2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex;flex-direction:column;gap:8px'>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Base currency</div><div class='metric'>{base_ccy} {flag_emoji_for_currency(base_ccy)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Target currency</div><div class='metric'>{target_ccy} {flag_emoji_for_currency(target_ccy)}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small'>Last update (UTC)</div><div class='metric'>{latest_data.get('date')}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ---------------------------
# Top 10 conversions table & bar chart
# ---------------------------
st.markdown("### 🔢 Top 10 currencies relative to base")
rates = latest_data.get("rates", {})
# Prepare dataframe
df_rates = pd.DataFrame(list(rates.items()), columns=["currency", "rate"])
# Add description if available
df_rates["description"] = df_rates["currency"].apply(lambda c: SYMBOLS.get(c, {}).get("description", ""))
# Sort by rate relative to base (we'll show the top 10 by trade-volume-like pick: major currencies)
major_list = ["USD", "EUR", "GBP", "JPY", "CNY", "AUD", "CAD", "CHF", "INR", "PKR", "SGD", "HKD", "NZD", "BRL", "ZAR", "TRY"]
# Choose top 10 available among major_list (base might be in them)
top_candidates = [c for c in major_list if c in rates]
if len(top_candidates) < 10:
    # fallback to top 10 by rate value
    top_df = df_rates.sort_values("rate", ascending=False).head(10)
else:
    top_df = pd.DataFrame({"currency": top_candidates})
    top_df["rate"] = top_df["currency"].apply(lambda c: rates.get(c))
    top_df["description"] = top_df["currency"].apply(lambda c: SYMBOLS.get(c, {}).get("description", ""))

# Show table
st.dataframe(top_df.reset_index(drop=True).assign(rate=lambda d: d["rate"].round(6)), height=220)

# Bar chart
fig_bar = px.bar(top_df, x="currency", y="rate", text=top_df["rate"].round(6),
                 title=f"Conversion rates relative to {base_ccy} (top currencies)",
                 labels={"rate": f"1 {base_ccy} → amount"})
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------
# Historical trend line chart
# ---------------------------
st.markdown("### 📈 Historical trend for selected pair")
timeseries_json = fetch_timeseries(base_ccy, target_ccy, days=days)
if not timeseries_json.get("success", True):
    st.warning("Historical data not available. Showing only latest rate.")
else:
    ts_rates = timeseries_json.get("rates", {})
    # Build series
    dates = sorted(ts_rates.keys())
    values = [ts_rates[d].get(target_ccy, np.nan) for d in dates]
    df_ts = pd.DataFrame({"date": pd.to_datetime(dates), "rate": values})
    # Line chart
    fig_line = px.line(df_ts, x="date", y="rate", title=f"{base_ccy}/{target_ccy} — Last {days} days",
                       markers=True, labels={"rate": f"1 {base_ccy} → {target_ccy}"})
    fig_line.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------
# Pie chart of distribution (top 6)
# ---------------------------
st.markdown("### 🥧 Currency distribution snapshot (Top 6)")
top6 = top_df.head(6).copy()
if top6.empty:
    st.info("No data for pie chart.")
else:
    fig_pie = px.pie(top6, names="currency", values="rate", title=f"Share by rate values vs {base_ccy}", hole=0.35)
    fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig_pie, use_container_width=False)

st.write("")

# ---------------------------
# Export & additional tools
# ---------------------------
st.markdown("### 💾 Export & Tools")
col_a, col_b, col_c = st.columns([2, 2, 2])
with col_a:
    # Build a results dataframe
    results_df = pd.DataFrame([{
        "requested_at_utc": datetime.utcnow().isoformat(),
        "base": base_ccy,
        "target": target_ccy,
        "amount": amount,
        "rate": used_rate,
        "converted": converted_value
    }])
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download conversion CSV", data=csv_bytes, file_name="conversion.csv", mime="text/csv")
with col_b:
    # Full rates CSV
    all_rates_df = pd.DataFrame(list(rates.items()), columns=["currency", "rate"])
    csv_all = all_rates_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download all rates CSV", data=csv_all, file_name=f"rates_{base_ccy}_{latest_data.get('date')}.csv", mime="text/csv")
with col_c:
    # Quick copy text
    st.code(f"{amount} {base_ccy} = {converted_value:,.4f} {target_ccy}  (Rate: {used_rate:.6f})")

st.write("")

# ---------------------------
# Error handling & tips
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("#### Tips & Notes")
st.markdown("""
- This app uses **exchangerate.host**, a free public API — **no API key required**.
- For the best UX, try phrases like: `Convert 250 USD to PKR` or `100 EUR to GBP`.
- Auto-refresh will rely on cached endpoints; manual refresh fetches fresh rates immediately.
- If you need more accurate currency-country flags or localized currency symbols, we can add `pycountry` and `babel` support.
""")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("<div class='small' style='margin-top:14px'>Made with ❤️ • Live rates from exchangerate.host • Built with Streamlit</div>", unsafe_allow_html=True)
