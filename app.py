import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import yfinance as yf
from github import Github 
import time
from FinMind.data import DataLoader
import twstock
import matplotlib.colors as mcolors
import io
import requests
from fugle_marketdata import RestClient

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="V32 戰情室 (Scout Mode)",
    layout="wide",
    page_icon="⚔️"
)

# --- 全域變數 ---
DATA_REPO = "gtty2003-ux/v32-auto-updater" 
DATA_FILE = "v32_dataset.csv"
HOLDING_REPO = "gtty2003-ux/v32-data"
HOLDINGS_FILE = "holdings.csv"

# --- 樣式設定 ---
st.markdown("""
    <style>
    .stDataFrame thead tr th {background-color: #ffebee !important; color: #b71c1c !important; font-weight: bold;}
    div[data-testid="stMetricValue"] {font-size: 24px; font-weight: bold;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    div[data-testid="stCaptionContainer"] {text-align: right; align-self: center; padding-top: 10px;}
    </style>
    """, unsafe_allow_html=True)

# --- 工具函數 ---
def get_taiwan_time_str(timestamp=None):
    """將 timestamp 轉為台灣時間字串"""
    tz = pytz.timezone('Asia/Taipei')
    if timestamp:
        # 將 epoch 時間視為 UTC，然後轉台灣時間
        dt = datetime.fromtimestamp(timestamp, pytz.utc).astimezone(tz)
    else:
        dt = datetime.now(tz)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_taiwan_time_iso():
    """取得台灣時間 (用於存檔)"""
    return datetime.now(pytz.timezone('Asia/Taipei')).strftime("%Y-%m-%d %H:%M:%S")

def make_pastel_cmap(hex_color):
    return mcolors.LinearSegmentedColormap.from_list("pastel_cmap", ["#ffffff", hex_color])

cmap_pastel_red = make_pastel_cmap("#ef9a9a")
cmap_pastel_blue = make_pastel_cmap("#90caf9")
cmap_pastel_green = make_pastel_cmap("#a5d6a7")

def color_surplus(val):
    if not isinstance(val, (int, float)): return ''
    return 'color: #d32f2f; font-weight: bold;' if val > 0 else ('color: #388e3c; font-weight: bold;' if val < 0 else 'color: black')

def color_action(val):
    val_str = str(val)
    if "🔴" in val_str or "停損" in val_str:
        return 'color: #ffffff; background-color: #d32f2f; font-weight: bold; padding: 5px; border-radius: 5px;' # 紅底白字
    elif "🟡" in val_str or "停利" in val_str:
        return 'color: #000000; background-color: #ffeb3b; font-weight: bold; padding: 5px; border-radius: 5px;' # 黃底黑字
    elif "🟢" in val_str or "續抱" in val_str:
        return 'color: #ffffff; background-color: #2e7d32; font-weight: bold; padding: 5px; border-radius: 5px;' # 綠底白字
    return ''

def color_risk(val):
    """地雷分顏色邏輯"""
    if not isinstance(val, (int, float)): return ''
    if val >= 60:
        return 'color: #ffffff; background-color: #d32f2f; font-weight: bold;' # 紅底 (高風險)
    elif val >= 30:
        return 'color: #000000; background-color: #ffeb3b; font-weight: bold;' # 黃底 (警戒)
    return 'color: #1b5e20; font-weight: bold;' # 綠字 (安全)

# --- 新增：大盤濾網模組 ---
@st.cache_data(ttl=3600) # 大盤一小時更新一次即可
def get_market_status():
    try:
        # 抓取台股大盤 (加權指數)
        twii = yf.Ticker("^TWII")
        hist = twii.history(period="6mo") # 抓半年數據
        
        if hist.empty: return None

        close = hist['Close']
        current_price = close.iloc[-1]
        
        # 計算均線
        ma20 = close.rolling(20).mean().iloc[-1] # 月線
        ma60 = close.rolling(60).mean().iloc[-1] # 季線 (生命線)
        
        # 判斷狀態
        status = "盤整/不明"
        signal = "🟡"
        
        # 邏輯判斷
        if current_price > ma60:
            if current_price > ma20:
                status = "多頭進攻 (安全)"
                signal = "🟢" 
            else:
                status = "多頭回檔 (警戒)"
                signal = "🟡"
        else:
            status = "空頭走勢 (危險 - 禁止買進)"
            signal = "🔴"
            
        return {
            'status': status,
            'signal': signal,
            'price': current_price,
            'ma60': ma60,
            'gap': (current_price - ma60) / ma60 * 100
        }
    except Exception as e:
        return None

# --- 資料讀取 ---
@st.cache_data(ttl=1800)
def load_data_from_github():
    try:
        token = st.secrets["general"]["GITHUB_TOKEN"]
        url = f"https://api.github.com/repos/{DATA_REPO}/contents/{DATA_FILE}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            df['Code'] = df['Code'].astype(str).str.strip()
            df['Date'] = pd.to_datetime(df['Date'])
            numeric_cols = ['ClosingPrice', 'OpeningPrice', 'HighestPrice', 'LowestPrice', 'TradeVolume']
            for c in numeric_cols:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            return df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# --- V32 運算邏輯 (放寬版: ADX>20) ---
def calculate_v32_score(df_group):
    if len(df_group) < 30: return None 
    df = df_group.sort_values('Date').reset_index(drop=True)
    
    # 基礎數據
    close = df['ClosingPrice']
    high = df['HighestPrice']
    low = df['LowestPrice']
    vol = df['TradeVolume']
    open_p = df['OpeningPrice']
    
    # 1. 計算均線
    ma5, ma20, ma60 = close.rolling(5).mean(), close.rolling(20).mean(), close.rolling(60).mean()
    
    # 2. 計算 ADX
    df['tr0'] = abs(high - low)
    df['tr1'] = abs(high - close.shift(1))
    df['tr2'] = abs(low - close.shift(1))
    tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    smooth_plus_dm = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    smooth_minus_dm = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    plus_di = 100 * (smooth_plus_dm / atr)
    minus_di = 100 * (smooth_minus_dm / atr)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/14, adjust=False).mean() 
    
    # 3. 準備當前數值
    i = -1 
    c_now = close.iloc[i]
    m20_now = ma20.iloc[i]
    v_now = vol.iloc[i]
    adx_now = adx.iloc[i]
    
    if pd.isna(c_now) or c_now == 0 or pd.isna(m20_now) or m20_now == 0 or pd.isna(adx_now): 
        return None

    # ==========================================
    # ⛔ 步驟 2: 放寬後的雙重濾網
    # ==========================================
    
    # 濾網 A: ADX 從 25 降為 20 (允許趨勢剛萌芽)
    if adx_now <= 20: 
        return None 
        
    # 濾網 B: 乖離率維持 15% (這是安全底線)
    bias_percentage = (c_now - m20_now) / m20_now * 100
    if bias_percentage >= 15:
        return None 

    # --- 激進派計分 (0分起跑) ---
    delta = close.diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    r_now = rsi.iloc[i]

    exp1, exp2 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    macd, signal = (exp1 - exp2), (exp1 - exp2).ewm(span=9, adjust=False).mean()
    
    vol_ma5, vol_ma20 = vol.rolling(5).mean(), vol.rolling(20).mean()
    high_20 = high.rolling(20).max()

    t_score = 0
    if c_now > m20_now: t_score += 10
    if m20_now > ma20.iloc[i-1]: t_score += 10
    if ma5.iloc[i] > m20_now > ma60.iloc[i]: t_score += 20
    if c_now >= high_20.iloc[i-1]: t_score += 30
    if r_now > 55: t_score += 15
    if macd.iloc[i] > signal.iloc[i]: t_score += 15

    v_score = 0
    if v_now > vol_ma20.iloc[i]: v_score += 20
    if v_now > vol_ma5.iloc[i]: v_score += 20
    if v_now > vol_ma20.iloc[i] * 1.5: v_score += 30
    if c_now > open_p.iloc[i] and v_now > vol.iloc[i-1]: v_score += 30
    
    final_score = (min(100, t_score) * 0.7) + (min(100, v_score) * 0.3)
    
    return {
        '技術分': min(100, t_score), 
        '量能分': min(100, v_score), 
        '攻擊分': final_score, 
        '收盤': c_now,
        '20MA': m20_now,
        'ADX': adx_now,
        '乖離率': bias_percentage
    }

@st.cache_data(ttl=1800)
def process_data():
    raw_df = load_data_from_github()
    if raw_df.empty: return pd.DataFrame(), pd.DataFrame(), "無法讀取數據"
    
    raw_df['Code_Str'] = raw_df['Code'].astype(str).str.strip()
    raw_df['Name_Str'] = raw_df['Name'].astype(str).str.strip()
    mask_common = (raw_df['Code_Str'].str.len() == 4) & (raw_df['Code_Str'].str.isdigit())
    mask_exclude = (
        raw_df['Code_Str'].str.startswith('28') |
        raw_df['Code_Str'].str.startswith('00') |
        raw_df['Code_Str'].str.startswith('91') |
        raw_df['Code_Str'].str.startswith('02') |
        raw_df['Name_Str'].str.contains('KY')   |
        raw_df['Name_Str'].str.contains('創')
    )
    raw_df = raw_df[mask_common & ~mask_exclude]

    results = []
    for code, group in raw_df.groupby('Code'):
        res = calculate_v32_score(group)
        if res:
            res.update({'代號': code, '名稱': group['Name'].iloc[-1]})
            results.append(res)
    return pd.DataFrame(results), raw_df, None

# --- 強化的即時報價模組 (Fugle API - Secrets 版) ---
def get_realtime_quotes_robust(code_list):
    realtime_data = {}
    try:
        api_key = st.secrets["general"]["FUGLE_API_KEY"]
    except:
        st.error("❌ 尚未設定 Fugle API Key！")
        return {}

    try:
        client = RestClient(api_key=api_key)
    except Exception as e:
        st.error(f"Fugle 連線失敗: {e}")
        return {}
    
    progress_bar = st.progress(0, text="🚀 富果引擎啟動中 (Fugle API)...")
    total = len(code_list)
    
    for idx, code in enumerate(code_list):
        clean_code = str(code).strip().split('.')[0]
        try:
            stock = client.stock
            q = stock.intraday.quote(symbol=clean_code)
            price = None
            if 'closePrice' in q: price = q['closePrice']
            elif 'lastPrice' in q: price = q['lastPrice']
            elif 'avgPrice' in q: price = q['avgPrice']
            if price: realtime_data[clean_code] = {'即時價': float(price)}
        except Exception: pass
        progress_bar.progress((idx + 1) / total)
    
    progress_bar.empty()
    return realtime_data

# --- 地雷股分數計算 ---
@st.cache_data(ttl=86400) 
def calculate_risk_factors(code):
    r1, r2, r3, r4 = 0, 0, 0, 0
    try:
        stock = yf.Ticker(f"{code}.TW")
        fin = stock.quarterly_financials
        bs = stock.quarterly_balance_sheet
        cf = stock.quarterly_cashflow
        if fin.empty: fin = stock.financials
        if bs.empty: bs = stock.balance_sheet
        if cf.empty: cf = stock.cashflow
        
        if not fin.empty and not cf.empty:
            try:
                ni = fin.loc['Net Income'].iloc[0]
                ocf_key = next((k for k in cf.index if 'Operating' in k), None)
                ocf = cf.loc[ocf_key].iloc[0] if ocf_key else ni 
                if ocf < ni:
                    ratio = (ni - ocf) / abs(ni) if ni != 0 else 0
                    r1 = min(30, ratio * 30) 
            except: pass

        if not fin.empty and not bs.empty and len(fin.columns) > 1:
            try:
                rev_now = fin.loc['Total Revenue'].iloc[0]
                rev_prev = fin.loc['Total Revenue'].iloc[1]
                rev_yoy = (rev_now - rev_prev) / rev_prev if rev_prev != 0 else 0
                inv_key = next((k for k in bs.index if 'Inventory' in k), None)
                inv_yoy = 0
                if inv_key:
                    inv_now = bs.loc[inv_key].iloc[0]
                    inv_prev = bs.loc[inv_key].iloc[1]
                    inv_yoy = (inv_now - inv_prev) / inv_prev if inv_prev != 0 else 0
                rec_key = next((k for k in bs.index if 'Receivables' in k), None)
                rec_yoy = 0
                if rec_key:
                    rec_now = bs.loc[rec_key].iloc[0]
                    rec_prev = bs.loc[rec_key].iloc[1]
                    rec_yoy = (rec_now - rec_prev) / rec_prev if rec_prev != 0 else 0
                gap = max(inv_yoy, rec_yoy) - rev_yoy
                if gap > 0: r2 = min(20, gap * 50)
            except: pass
        
        if not bs.empty:
            try:
                cash_key = next((k for k in bs.index if 'Cash' in k), None)
                rec_key = next((k for k in bs.index if 'Receivables' in k), None)
                liab_key = next((k for k in bs.index if 'Current Liabilities' in k), None)
                cash = bs.loc[cash_key].iloc[0] if cash_key else 0
                rec = bs.loc[rec_key].iloc[0] if rec_key else 0
                liab = bs.loc[liab_key].iloc[0] if liab_key else 1
                qr = (cash + rec) / liab
                if qr <= 0.5: r3 = 20
                elif qr >= 1.5: r3 = 0
                else: r3 = 20 * (1.5 - qr)
            except: pass

        try:
            url = f"https://histock.tw/stock/large.aspx?no={code}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers)
            dfs = pd.read_html(io.StringIO(r.text))
            pledge_ratio = 0
            for df in dfs:
                if '質押比例' in df.columns:
                    val = str(df['質押比例'].iloc[0]).replace('%', '')
                    pledge_ratio = float(val)
                    break
            r4 = min(30, pledge_ratio * 0.4)
        except: pass

        total = r1 + r2 + r3 + r4
        detail_str = f"現:{int(r1)} 膨:{int(r2)} 償:{int(r3)} 質:{int(r4)}"
        return total, detail_str
    except Exception:
        return 0, "無數據"

def get_risk_analysis_batch(code_list):
    risk_data = {}
    progress_bar = st.progress(0)
    total = len(code_list)
    for idx, code in enumerate(code_list):
        score, detail = calculate_risk_factors(code)
        risk_data[code] = {'地雷分': score, '風險細節': detail}
        progress_bar.progress((idx + 1) / total)
        time.sleep(0.5)
    progress_bar.empty()
    return pd.DataFrame.from_dict(risk_data, orient='index').reset_index().rename(columns={'index': '代號'})

# --- 籌碼分析 ---
def get_chip_analysis(symbol_list):
    chip_data = []
    p_bar = st.progress(0, text="連線至 HiStock 資料庫...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    total = len(symbol_list)
    for i, symbol in enumerate(symbol_list):
        p_bar.progress((i + 1) / total, text=f"分析中: {symbol}")
        try:
            url = f"https://histock.tw/stock/chips.aspx?no={symbol}"
            r = requests.get(url, headers=headers, timeout=8)
            dfs = pd.read_html(io.StringIO(r.text))
            target_df = None
            for df in dfs:
                if '日期' in df.columns and '外資' in df.columns:
                    target_df = df
                    break
            if target_df is not None and not target_df.empty:
                latest = target_df.iloc[0]
                def clean_num(val):
                    try: return int(str(val).replace(',', '').replace('+', ''))
                    except: return 0
                f_buy = clean_num(latest['外資'])
                t_buy = clean_num(latest['投信'])
                
                status_str = ""
                if t_buy > 0: status_str += "🔴 投信買 "
                elif t_buy < 0: status_str += "🟢 投信賣 "
                if f_buy > 1000: status_str += "🔥 外資大買 "
                elif f_buy < -1000: status_str += "🧊 外資倒貨 "
                
                if t_buy > 0 and f_buy > 0: tag = "🚀 土洋合買"
                elif t_buy > 0 and f_buy < 0: tag = "⚔️ 土洋對作(信)"
                elif t_buy < 0 and f_buy > 0: tag = "⚔️ 土洋對作(外)"
                elif t_buy < 0 and f_buy < 0: tag = "☠️ 主力棄守"
                else: tag = "🟡 一般輪動"
                
                final_status = f"{tag} | {status_str}" if status_str else tag
                chip_data.append({'代號': symbol, '投信(張)': t_buy, '外資(張)': f_buy, '主力動向': final_status})
            else:
                chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': '🟡 無數據'})
        except Exception as e:
            chip_data.append({'代號': symbol, '投信(張)': 0, '外資(張)': 0, '主力動向': '❌ Error'})
        time.sleep(1.0)
    p_bar.progress(100, text="分析完成")
    time.sleep(0.5)
    p_bar.empty()
    return pd.DataFrame(chip_data)

# --- 庫存管理 ---
def load_holdings():
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        df = pd.read_csv(g.get_repo(HOLDING_REPO).get_contents(HOLDINGS_FILE).download_url)
        df['股票代號'] = df['股票代號'].astype(str).apply(lambda x: x.split('.')[0] if '.' in x else x)
        return df[['股票代號', '買入均價', '持有股數']]
    except: return pd.DataFrame(columns=["股票代號", "買入均價", "持有股數"])

def save_holdings(df):
    try:
        g = Github(st.secrets["general"]["GITHUB_TOKEN"])
        repo = g.get_repo(HOLDING_REPO)
        csv_content = df.to_csv(index=False)
        contents = repo.get_contents(HOLDINGS_FILE)
        repo.update_file(contents.path, f"Update {get_taiwan_time_iso()}", csv_content, contents.sha)
    except: pass

# --- Tab 1 & 2 表格渲染 (放寬顯示門檻: 60分以上) ---
def display_v32_tables(df, price_limit, suffix):
    # 修改點：這裡將顯示門檻從 80 降為 60，讓 B 級股也能顯示
    filtered = df[(df['收盤'] <= price_limit) & (df['攻擊分'] >= 60)].sort_values('攻擊分', ascending=False)
    
    if filtered.empty: 
        st.warning("目前無符合標準 (ADX>20 & 乖離<15%) 的標的，建議空手觀望。")
        return

    df_s_pre = filtered[(filtered['攻擊分'] >= 90)].head(10)
    df_a_pre = filtered[(filtered['攻擊分'] >= 80) & (filtered['攻擊分'] < 90)].head(10)
    df_b_pre = filtered[(filtered['攻擊分'] >= 60) & (filtered['攻擊分'] < 80)].head(10) # 新增 B 級
    target_codes = pd.concat([df_s_pre, df_a_pre, df_b_pre])['代號'].tolist()

    c_scan, c_risk, c_update, c_info = st.columns([1, 1, 1, 1.5])
    chip_key = f"chip_data_{suffix}"
    risk_key = f"risk_data_{suffix}"

    with c_scan:
        if st.button(f"🚀 籌碼掃描", key=f"scan_{suffix}"):
            chip_df = get_chip_analysis(target_codes)
            st.session_state[chip_key] = chip_df 
    with c_risk:
        if st.button(f"💣 地雷檢測", key=f"risk_{suffix}"):
            with st.spinner("正在進行深度財報與質押掃描..."):
                risk_df = get_risk_analysis_batch(target_codes)
                st.session_state[risk_key] = risk_df 
    with c_update:
        now = time.time()
        time_diff = now - st.session_state.get('last_update_time', 0)
        btn_label = "🔄 更新即時價"
        btn_disabled = False
        if time_diff < 60:
            btn_label = f"⏳ 冷卻 ({int(60 - time_diff)}s)"
            btn_disabled = True
        if st.button(btn_label, disabled=btn_disabled, key=f"update_{suffix}", type="primary"):
            with st.spinner(f"🚀 同步 Top {len(target_codes)} 檔股價..."):
                fresh_quotes = get_realtime_quotes_robust(target_codes)
                current_quotes = st.session_state.get('realtime_quotes', {})
                current_quotes.update(fresh_quotes)
                st.session_state['realtime_quotes'] = current_quotes
                st.session_state['last_update_time'] = time.time()
                st.toast(f"✅ 更新成功！", icon="🔄")
                time.sleep(1)
                st.rerun()
    with c_info:
        if st.session_state.get('last_update_time', 0) > 0:
            tw_time = get_taiwan_time_str(st.session_state['last_update_time'])
            st.caption(f"🕒 更新: {tw_time}")

    if chip_key in st.session_state:
        filtered = pd.merge(filtered, st.session_state[chip_key], on='代號', how='left')
    if risk_key in st.session_state:
        filtered = pd.merge(filtered, st.session_state[risk_key], on='代號', how='left')

    saved_quotes = st.session_state.get('realtime_quotes', {})
    filtered['即時價'] = filtered['代號'].map(lambda x: saved_quotes.get(x, {}).get('即時價', np.nan))
    filtered['即時價'] = filtered['即時價'].fillna(filtered['收盤'])

    base_cols = ['代號','名稱','即時價','技術分','量能分','攻擊分']
    if 'ADX' in filtered.columns: base_cols.append('ADX')
    if '乖離率' in filtered.columns: base_cols.append('乖離率')
    if '主力動向' in filtered.columns: base_cols += ['主力動向', '投信(張)', '外資(張)']
    if '地雷分' in filtered.columns: base_cols += ['地雷分', '風險細節']

    fmt = {'即時價':'{:.2f}', '攻擊分':'{:.1f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '外資(張)': '{:,.0f}', '投信(張)': '{:,.0f}', '地雷分':'{:.0f}', 'ADX':'{:.1f}', '乖離率':'{:.1f}%'}

    # 顯示三個區塊
    for title, score_range in [
        ("👑 S 級主力區 (90分以上)", (90, 100)),
        ("🚀 A 級蓄勢區 (80-90分)", (80, 90)),
        ("👀 B 級觀察區 (60-80分)", (60, 80)) # 新增
    ]:
        st.subheader(title)
        sub = filtered[(filtered['攻擊分'] >= score_range[0]) & (filtered['攻擊分'] < score_range[1] + 0.1)].head(10)
        
        if not sub.empty:
            st.dataframe(sub[base_cols].style.format(fmt)
                         .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red, vmin=60, vmax=100)
                         .background_gradient(subset=['技術分'], cmap=cmap_pastel_blue, vmin=0, vmax=100)
                         .background_gradient(subset=['量能分'], cmap=cmap_pastel_green, vmin=0, vmax=100)
                         .map(color_risk, subset=['地雷分'] if '地雷分' in sub.columns else []), 
                         hide_index=True, use_container_width=True)
        else: 
            st.caption("無符合標的")
        st.divider()

# --- 新增：個股搜尋專用顯示函式 ---
def display_single_stock_search(df, target_code):
    row = df[df['代號'] == target_code].copy()
    if row.empty:
        st.warning(f"⚠️ 資料庫中找不到代號 {target_code}，或該股 ADX<20 被剔除。")
        return

    search_key_chip = f"search_chip_{target_code}"
    search_key_risk = f"search_risk_{target_code}"
    
    col_input, col_btn = st.columns([3, 2])
    with col_btn:
        if st.button("🔎 立即詳細診斷", key=f"btn_search_{target_code}", type="primary"):
            with st.spinner(f"正在深度分析 {target_code} ..."):
                q = get_realtime_quotes_robust([target_code])
                st.session_state['realtime_quotes'].update(q)
                c = get_chip_analysis([target_code])
                st.session_state[search_key_chip] = c
                r = get_risk_analysis_batch([target_code])
                st.session_state[search_key_risk] = r
                st.rerun()

    if search_key_chip in st.session_state:
        row = pd.merge(row, st.session_state[search_key_chip], on='代號', how='left')
    if search_key_risk in st.session_state:
        row = pd.merge(row, st.session_state[search_key_risk], on='代號', how='left')
        
    saved_quotes = st.session_state.get('realtime_quotes', {})
    row['即時價'] = saved_quotes.get(target_code, {}).get('即時價', np.nan)
    row['即時價'] = row['即時價'].fillna(row['收盤'])

    base_cols = ['代號','名稱','即時價','技術分','量能分','攻擊分', 'ADX', '乖離率']
    if '主力動向' in row.columns: base_cols += ['主力動向', '投信(張)', '外資(張)']
    if '地雷分' in row.columns: base_cols += ['地雷分', '風險細節']

    fmt = {'即時價':'{:.2f}', '攻擊分':'{:.1f}', '技術分':'{:.0f}', '量能分':'{:.0f}', '外資(張)': '{:,.0f}', '投信(張)': '{:,.0f}', '地雷分':'{:.0f}', 'ADX':'{:.1f}', '乖離率':'{:.1f}%'}

    st.markdown(f"### 🎯 {target_code} 分析結果")
    st.dataframe(row[base_cols].style.format(fmt)
                 .background_gradient(subset=['攻擊分'], cmap=cmap_pastel_red, vmin=0, vmax=100) 
                 .map(color_risk, subset=['地雷分'] if '地雷分' in row.columns else [])
                 .map(color_action, subset=['主力動向'] if '主力動向' in row.columns else []), 
                 hide_index=True, use_container_width=True)

# --- 主程式 ---
def main():
    st.title("⚔️ V32 戰情室 (Scout Mode)")
    market = get_market_status()
    if market:
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            if "危險" in market['status']: st.error(f"{market['signal']} **大盤濾網：{market['status']}**")
            else: st.info(f"{market['signal']} **大盤濾網：{market['status']}**")
        with c2: st.metric("加權指數", f"{market['price']:,.0f}", f"{market['gap']:.2f}% (距季線)")
        with c3: st.metric("季線 (60MA)", f"{market['ma60']:,.0f}")
        st.divider() 
        
    if 'inventory' not in st.session_state: st.session_state['inventory'] = load_holdings()
    if 'realtime_quotes' not in st.session_state: st.session_state['realtime_quotes'] = {}
    if 'last_update_time' not in st.session_state: st.session_state['last_update_time'] = 0
    
    with st.spinner("讀取核心資料..."):
        v32_df, raw_df, err = process_data()
    
    tab_80, tab_50, tab_search, tab_inv = st.tabs(["💰 80元以下推薦", "🪙 50元以下推薦", "🔍 個股診斷", "💼 庫存管理"])

    with tab_80:
        if not v32_df.empty: display_v32_tables(v32_df.copy(), 80, "80")
        else: st.warning("目前無符合標準的標的。")

    with tab_50:
        if not v32_df.empty: display_v32_tables(v32_df.copy(), 50, "50")
        else: st.warning("目前無符合標準的標的。")

    with tab_search:
        st.subheader("🔍 個股 V32 體檢室")
        c1, c2 = st.columns([1, 3])
        with c1: search_input = st.text_input("輸入股票代號", placeholder="例如: 2330", max_chars=4)
        if search_input:
            clean_code = search_input.strip()
            if not v32_df.empty: display_single_stock_search(v32_df.copy(), clean_code)
            else: st.error("資料尚未載入")

    with tab_inv:
        st.subheader("📝 庫存交易管理")
        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            now = time.time()
            time_diff = now - st.session_state.get('last_update_time', 0)
            btn_label = "🔄 更新即時股價"
            btn_disabled = False
            if time_diff < 60:
                btn_label = f"⏳ 冷卻中 ({int(60 - time_diff)}s)"
                btn_disabled = True
            if st.button(btn_label, disabled=btn_disabled, type="primary", key="btn_inv_update"):
                if not st.session_state['inventory'].empty:
                    with st.spinner("🚀 同步庫存股價..."):
                        codes = st.session_state['inventory']['股票代號'].tolist()
                        fresh_quotes = get_realtime_quotes_robust(codes)
                        current_quotes = st.session_state.get('realtime_quotes', {})
                        current_quotes.update(fresh_quotes)
                        st.session_state['realtime_quotes'] = current_quotes
                        st.session_state['last_update_time'] = time.time()
                        st.toast(f"✅ 更新成功！", icon="💼")
                        time.sleep(1)
                        st.rerun()
        
        with col_info:
            if st.session_state.get('last_update_time', 0) > 0:
                tw_time = get_taiwan_time_str(st.session_state['last_update_time'])
                st.caption(f"🕒 台灣時間最後更新: {tw_time}")

        name_map = {}
        if not raw_df.empty: name_map = dict(zip(raw_df['Code'], raw_df['Name']))
        
        score_map, ma20_map, filtered_in_codes = {}, {}, []
        if not v32_df.empty:
            score_map = v32_df.set_index('代號')['攻擊分'].to_dict()
            filtered_in_codes = v32_df['代號'].tolist()
            if '20MA' in v32_df.columns: ma20_map = v32_df.set_index('代號')['20MA'].to_dict()
            else: ma20_map = {code: 0 for code in v32_df['代號']}

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📥 **買入**")
            edited_buy = st.data_editor(pd.DataFrame([{"股票代號": "", "持有股數": 1000, "買入均價": 0.0}]), num_rows="dynamic", key="buy_in", hide_index=True)
        with c2:
            st.markdown("##### 📤 **賣出**")
            edited_sell = st.data_editor(pd.DataFrame([{"股票代號": "", "持有股數": 1000}]), num_rows="dynamic", key="sell_out", hide_index=True)
        
        if st.button("💾 執行交易", type="primary"):
            inv = st.session_state['inventory'].copy()
            for _, r in edited_buy.iterrows():
                code = str(r['股票代號']).strip().split('.')[0]
                if code and r['持有股數'] > 0:
                    match = inv[inv['股票代號'] == code]
                    if not match.empty:
                        idx = match.index[0]
                        total_shares = inv.at[idx, '持有股數'] + r['持有股數']
                        inv.at[idx, '買入均價'] = round(((inv.at[idx, '買入均價'] * inv.at[idx, '持有股數']) + (r['買入均價'] * r['持有股數'])) / total_shares, 2)
                        inv.at[idx, '持有股數'] = total_shares
                    else:
                        inv = pd.concat([inv, pd.DataFrame([{'股票代號': code, '持有股數': r['持有股數'], '買入均價': r['買入均價']}])], ignore_index=True)
            for _, r in edited_sell.iterrows():
                code = str(r['股票代號']).strip().split('.')[0]
                if code:
                    inv = inv[~((inv['股票代號'] == code) & (inv['持有股數'] <= r['持有股數']))]
                    mask = inv['股票代號'] == code
                    if mask.any(): inv.loc[mask, '持有股數'] -= r['持有股數']
            st.session_state['inventory'] = inv
            save_holdings(inv)
            st.rerun()

        st.divider()
        st.subheader("⚡ 快速操作區")
        c1, c2 = st.columns(2)
        current_inv_codes = []
        if not st.session_state['inventory'].empty:
            current_inv_codes = st.session_state['inventory']['股票代號'].unique().tolist()

        with c1:
            st.markdown("##### 📉 個股清倉")
            to_sell_all = st.multiselect("選擇要全數賣出的股票", options=current_inv_codes)
            if st.button("💥 執行個股清倉", type="primary", disabled=not to_sell_all):
                inv = st.session_state['inventory'].copy()
                inv = inv[~inv['股票代號'].isin(to_sell_all)]
                st.session_state['inventory'] = inv
                save_holdings(inv)
                st.toast(f"已清空: {', '.join(to_sell_all)}", icon="💥")
                time.sleep(1)
                st.rerun()

        with c2:
            st.markdown("##### 🧨 重置帳戶")
            st.warning("注意：此操作將刪除所有庫存紀錄")
            if st.button("🗑️ 全部清空 (刪除所有庫存)", type="secondary"):
                inv = pd.DataFrame(columns=["股票代號", "買入均價", "持有股數"])
                st.session_state['inventory'] = inv
                save_holdings(inv)
                st.toast("已清空所有庫存！", icon="🗑️")
                time.sleep(1)
                st.rerun()

        st.divider()
        if not st.session_state['inventory'].empty:
            inv_df = st.session_state['inventory'].copy()
            saved_quotes = st.session_state.get('realtime_quotes', {})
            res = []
            for _, r in inv_df.iterrows():
                code = str(r['股票代號'])
                curr = saved_quotes.get(code, {}).get('即時價', r['買入均價'])
                if (curr == 0 or curr == r['買入均價']) and not raw_df.empty:
                      backup_data = raw_df[raw_df['Code']==code]
                      if not backup_data.empty: curr = backup_data['ClosingPrice'].values[0]

                buy_price = r['買入均價']
                qty = r['持有股數']
                pl = (curr - buy_price) * qty
                roi = (pl / (buy_price * qty) * 100) if buy_price > 0 else 0
                sc = score_map.get(code, 0)
                ma20 = ma20_map.get(code, 0)
                passed_filter = code in filtered_in_codes
                
                if curr < ma20 and ma20 > 0: action = f"🔴 停損/清倉 (破月線 {ma20:.1f})"
                elif not passed_filter: action = "⚠️ 趨勢不明/過熱 (濾網剔除)"
                elif sc >= 60: action = f"🟢 續抱 (攻擊分 {sc:.0f})"
                else: action = f"🟡 動能偏弱 (攻擊分 {sc:.0f})"

                res.append({'代號': code, '名稱': name_map.get(code, code), '持有張數': int(qty // 1000), '買入均價': buy_price, '即時價': curr, '損益': pl, '報酬率%': roi, '攻擊分': sc, '建議操作': action})
            
            df_res = pd.DataFrame(res)
            c1, c2, c3 = st.columns(3)
            c1.metric("總成本", f"${(df_res['買入均價']*(inv_df['持有股數'])).sum():,.0f}")
            c2.metric("總損益", f"${df_res['損益'].sum():,.0f}", delta=f"{df_res['損益'].sum():,.0f}")
            c3.metric("總市值", f"${(df_res['即時價']*(inv_df['持有股數'])).sum():,.0f}")
            
            def color_sniper_action(val):
                val_str = str(val)
                if "🔴" in val_str: return 'color: #ffffff; background-color: #d32f2f; font-weight: bold; padding: 5px; border-radius: 5px;'
                elif "🟢" in val_str: return 'color: #ffffff; background-color: #2e7d32; font-weight: bold; padding: 5px; border-radius: 5px;'
                elif "⚠️" in val_str: return 'color: #000000; background-color: #e0e0e0; font-weight: bold; padding: 5px; border-radius: 5px;'
                elif "🟡" in val_str: return 'color: #000000; background-color: #ffeb3b; font-weight: bold; padding: 5px; border-radius: 5px;'
                return ''

            st.dataframe(df_res[['代號', '名稱', '持有張數', '買入均價', '即時價', '攻擊分', '報酬率%', '損益', '建議操作']].style.format({'買入均價':'{:.2f}', '即時價':'{:.2f}', '損益':'{:+,.0f}', '報酬率%':'{:+.2f}%', '攻擊分':'{:.1f}'}).map(color_surplus, subset=['損益','報酬率%']).map(color_sniper_action, subset=['建議操作']), use_container_width=True, hide_index=True)
        else: st.info("目前無庫存。")

if __name__ == "__main__":
    main()
