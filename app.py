# app.py - AI選股系統 Streamlit 網站程式碼

# -----------------------------------------------------------
# 步驟 1: 匯入必要的套件 (Streamlit 與分析套件)
# -----------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf 
import pandas_ta as ta 

# -----------------------------------------------------------
# 步驟 2: 設定常數與對應表
# -----------------------------------------------------------
PRICE_LIMIT = 50          
STOCK_NAMES_MAP = {
    # 新增台灣50部分常用的中文名稱對應 (可自行擴充)
    '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2308': '台達電', '2881': '富邦金', 
    '2882': '國泰金', '1301': '台塑', '1303': '南亞', '6505': '台塑化', '1216': '統一', 
    '2891': '中信金', '2886': '兆豐金', '3008': '大立光', '2207': '和泰車', '3711': '日月光投控',
    '2303': '聯電', '1101': '台泥', '2002': '中鋼', '1314': '中石化', '2409': '友達', 
    '3231': '緯創', '1710': '東聯', '1402': '遠東新', '2498': '宏達電', '2501': '國建', 
    '2888': '新光金' 
}
    
# V16 修正：擴大選股池至台灣50成分股 (範例清單，約50支)
SAMPLE_STOCKS = [
    '2330.TW', '2317.TW', '2454.TW', '2308.TW', '2881.TW', '2882.TW', '1303.TW', '1301.TW', '6505.TW', '1216.TW', 
    '2891.TW', '2886.TW', '3008.TW', '2207.TW', '3711.TW', '2303.TW', '1101.TW', '2002.TW', '1314.TW', '2409.TW', 
    '3231.TW', '1710.TW', '1402.TW', '2498.TW', '2501.TW', '2888.TW', '2049.TW', '2382.TW', '2801.TW', '2834.TW', 
    '4904.TW', '5871.TW', '8427.TW', '2892.TW', '2915.TW', '3034.TW', '3045.TW', '3481.TW', '4938.TW', '5880.TW', 
    '6415.TW', '6477.TW', '8046.TW', '8210.TW', '9904.TW', '2301.TW', '2353.TW', '2412.TW', '2603.TW', '2610.TW' 
]
MAX_SCORE = 1.00 # 模型最高總分 (用於轉換為 100 分制)


@st.cache_data(ttl=dt.timedelta(hours=24))
def get_and_prepare_data(start_date, end_date, stocks):
    
    final_data_list = []
    
    for stock_id in stocks:
        try:
            df = yf.download(stock_id, start=start_date, end=end_date, progress=False)
            if df.empty:
                continue
            
            # 數據結構修正
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
            df.columns = df.columns.str.replace(' ', '')
            df.index.name = 'date'
            df = df.reset_index()
            df['stock_id'] = stock_id
            df['stock_code'] = stock_id.split('.')[0]
            
            # 1. 計算技術指標 (確保 BBANDS 週期穩定)
            df.ta.rsi(close='Close', append=True)
            df.ta.macd(close='Close', append=True)
            df.ta.stoch(close='Close', append=True) 
            df.ta.bbands(close='Close', length=14, std=2.0, append=True) 
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Volume_MA20']
            
            latest_data = df.tail(1).copy()
            final_data_list.append(latest_data) 
            
        except Exception:
            continue

    if not final_data_list:
        return pd.DataFrame()
        
    df_final = pd.concat(final_data_list, ignore_index=True)
    # V16: 移除價格過高的股票，但價格限制不適用於高價權值股，可選擇移除或提高限制。
    # 這裡我們先維持 <= 50 元的篩選，讓模型只在低價股中尋找機會。
    candidate_stocks = df_final[df_final['Close'] <= PRICE_LIMIT].reset_index(drop=True)
    
    return candidate_stocks

def run_simple_momentum_model(input_data):
    
    if input_data.empty:
        return pd.DataFrame()
        
    df = input_data.copy()
    
    # V15 修正：動態獲取 BBANDS 欄位名稱
    bb_cols = [col for col in df.columns if col.startswith('BBU_')]
    bb_upper_band = bb_cols[0] if bb_cols else None
    
    # V16: 初始化所有分數欄位，避免 KeyError 或 NaN 錯誤
    df['Score_MACD'] = 0.0
    df['Score_KDJ_Cross'] = 0.0
    df['Score_RSI_Strong'] = 0.0
    df['Score_Volume'] = 0.0
    df['Score_BB_Breakout'] = 0.0
    df['Score_RSI_Oversold'] = 0.0
    
    # ----------------------------------------------------------------
    # V16: 短線動能模型權重 (總分 1.00) 
    # ----------------------------------------------------------------
    
    # 1. 趨勢確認 (MACD > 0) - 0.20
    df['Score_MACD'] = np.where(df['MACDh_12_26_9'] > 0, 0.20, 0)
    
    # 2. 短線發動點 (KDJ 金叉: K > D 且 D < 50) - 0.25
    df['Score_KDJ_Cross'] = np.where((df['STOCHk_14_3_3'] > df['STOCHd_14_3_3']) & (df['STOCHd_14_3_3'] < 50), 0.25, 0)
    
    # 3. 強勢區間 (RSI 50~70) - 0.20
    df['Score_RSI_Strong'] = np.where((df['RSI_14'] > 50) & (df['RSI_14'] < 70), 0.20, 0)
    
    # 4. 成交量動能 (Vol > MA20) - 0.15
    df['Score_Volume'] = np.where(df['Vol_Ratio'] > 1.0, 0.15, 0)
    
    # 5. 波動率擴張 (股價突破布林通道上軌) - 0.15
    if bb_upper_band:
        df['Score_BB_Breakout'] = np.where(df['Close'] > df[bb_upper_band], 0.15, 0)
    
    # 6. 反彈潛力 (RSI 超賣) - 0.05
    df['Score_RSI_Oversold'] = np.where(df['RSI_14'] < 30, 0.05, 0)
    
    # 計算總分
    df['AI_Score'] = df['Score_MACD'] + df['Score_KDJ_Cross'] + df['Score_RSI_Strong'] + df['Score_Volume'] + df['Score_BB_Breakout'] + df['Score_RSI_Oversold']
    
    # 生成推薦理由
    def generate_reason(row):
        reasons = []
        if row['Score_MACD'] > 0: reasons.append('MACD上揚, 趨勢確認')
        if row['Score_KDJ_Cross'] > 0: reasons.append('KDJ低檔金叉, 短線動能啟動')
        if row['Score_RSI_Strong'] > 0: reasons.append('RSI處於強勢區間(50~70)')
        if row['Score_Volume'] > 0: reasons.append('成交量突破20日均量, 買盤積極')
        if row['Score_BB_Breakout'] > 0: reasons.append('股價突破布林通道上軌, 波動率擴張')
        if row['Score_RSI_Oversold'] > 0: reasons.append('RSI超賣, 具備反彈潛力')
            
        return '; '.join(reasons) if reasons else '技術指標中性'

    df['推薦理由'] = df.apply(generate_reason, axis=1)
    
    # 排序並取出 Top 5
    top_stocks = df.sort_values(by='AI_Score', ascending=False)
    
    # 確保只篩選分數大於 0 的股票，防止推薦無訊號的股票
    top_stocks = top_stocks[top_stocks['AI_Score'] > 0]
    
    final_recommendations = top_stocks.head(5)[['stock_code', 'Close', 'AI_Score', '推薦理由']]
    
    # 增加中文名稱欄位
    final_recommendations['股票名稱'] = final_recommendations['stock_code'].apply(
        lambda x: STOCK_NAMES_MAP.get(x, f'代碼{x}')
    )
    
    return final_recommendations

# -----------------------------------------------------------
# 步驟 3: Streamlit 網頁主體
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title='AI 短期波段選股系統', layout='wide')
    st.title('📈 AI 短期波段選股系統')
    st.markdown(f'**分析模型:** 短線動能強化模型 (KDJ, BBANDS, MACD, RSI, Vol) | **價格限制:** ≤ {PRICE_LIMIT} 元. **選股池:** 台灣50成分股') # V16 調整說明
    
    # --- 新增：立即更新按鈕 ---
    if st.button('🔄 立即手動更新數據 (清除緩存)'):
        st.cache_data.clear()
        st.success('數據緩存已清除, 正在重新獲取資料...')
        st.rerun()
    # --------------------------
    
    # 設定日期範圍
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=120) 
    
    # 獲取並處理數據
    with st.spinner('🔄 正在獲取數據並運行分析模型 (約 50 檔)...'):
        processed_data = get_and_prepare_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), SAMPLE_STOCKS)
    
    st.info(f'數據更新時間: {end_date.strftime('%Y-%m-%d')} (數據每 24 小時自動更新)')

    if processed_data.empty:
        st.error('❌ 無法獲取有效數據. 請檢查數據源或網路連接.')
        return

    # 運行分析模型
    final_recommendations = run_simple_momentum_model(processed_data)

    st.header('🏆 本日 Top 5 推薦清單')
    
    # --- 輸出格式化 ---
    MAX_SCORE_VALUE = 1.00
    # 1. 轉換分數為 100 分制 (基於 MAX_SCORE 1.00)
    final_recommendations['AI_Score'] = final_recommendations['AI_Score'] * 100 / MAX_SCORE_VALUE
    
    # 2. 調整輸出順序和欄位名稱
    final_recommendations = final_recommendations[['stock_code', '股票名稱', 'Close', 'AI_Score', '推薦理由']]
    final_recommendations = final_recommendations.rename(columns={
        'stock_code': '股票代碼',
        'Close': '當日收盤價 (元)', 
        'AI_Score': '分析分數'
    })
    
    # 3. 格式化輸出
    final_recommendations['當日收盤價 (元)'] = final_recommendations['當日收盤價 (元)'].apply(lambda x: f'{x:,.2f}')
    final_recommendations['分析分數'] = final_recommendations['分析分數'].apply(lambda x: f'{x:.1f}')
    # --------------------

    # 使用 Streamlit 顯示表格
    if final_recommendations.empty:
        st.warning('⚠️ 警告：在價格限制 (≤ 50 元) 下，選股池中沒有股票符合任何動能信號。')
    else:
        st.dataframe(final_recommendations, use_container_width=True, hide_index=True)

    # 更新備註說明
    markdown_notes = (
        '---\n'
        '**備註說明:**\n'
        f'- 分析分數為 [0, {int(MAX_SCORE*100)}], 分數越高, 技術面訊號越強.\n'
        f'- 選股池已擴大至台灣50成分股，並篩選股價 ≤ {PRICE_LIMIT} 元的標的。'
    )
    st.markdown(markdown_notes)

if __name__ == '__main__':
    main()