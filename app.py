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
    '2303': '聯電', '1101': '台泥', '2002': '中鋼', '1314': '中石化', 
    '2409': '友達', '3231': '緯創', '3008': '大立光', '1710': '東聯', 
    '1402': '遠東新', '2498': '宏達電', '2501': '國建', '2891': '中信金',
    '2888': '新光金' 
}
SAMPLE_STOCKS = ['2303.TW', '1101.TW', '2002.TW', '1314.TW', '2409.TW', '3231.TW', '3008.TW', '1710.TW', '1402.TW', '2498.TW', '2501.TW', '2891.TW']


@st.cache_data(ttl=datetime.timedelta(hours=24))
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
            df['stock_name_en'] = stock_id.split('.')[0]
            
            # 1. 計算技術指標
            df.ta.rsi(close='Close', append=True)
            df.ta.macd(close='Close', append=True)
            df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
            df['Vol_Ratio'] = df['Volume'] / df['Volume_MA20']
            
            latest_data = df.tail(1).copy()
            final_data_list.append(latest_data) 
            
        except Exception:
            continue

    if not final_data_list:
        return pd.DataFrame()
        
    df_final = pd.concat(final_data_list, ignore_index=True)
    candidate_stocks = df_final[df_final['Close'] <= PRICE_LIMIT].set_index('stock_id')
    
    return candidate_stocks

def run_simple_momentum_model(input_data):
    
    if input_data.empty:
        return pd.DataFrame()
        
    df = input_data.copy()
    
    # 計算分數
    df['Score_MACD'] = np.where(df['MACDh_12_26_9'] > 0, 0.3, 0)
    df['Score_RSI_Strong'] = np.where((df['RSI_14'] > 50) & (df['RSI_14'] < 70), 0.2, 0)
    df['Score_RSI_Oversold'] = np.where(df['RSI_14'] < 30, 0.1, 0)
    df['Score_Volume'] = np.where(df['Vol_Ratio'] > 1.0, 0.15, 0)
    
    df['AI_Score'] = df['Score_MACD'] + df['Score_RSI_Strong'] + df['Score_RSI_Oversold'] + df['Score_Volume']
    
    # 生成推薦理由
    def generate_reason(row):
        reasons = []
        if row['Score_MACD'] > 0: reasons.append('MACD上揚, 趨勢轉強')
        if row['Score_RSI_Strong'] > 0: reasons.append('RSI處於強勢區間(50~70)')
        if row['Score_RSI_Oversold'] > 0: reasons.append('RSI超賣, 具備反彈潛力')
        if row['Score_Volume'] > 0: reasons.append('成交量突破20日均量, 買盤積極')
            
        return '; '.join(reasons) if reasons else '技術指標中性'

    df['推薦理由'] = df.apply(generate_reason, axis=1)
    
    # 排序並取出 Top 5
    top_stocks = df.sort_values(by='AI_Score', ascending=False)
    
    final_recommendations = top_stocks.head(5)[['stock_name_en', 'Close', 'AI_Score', '推薦理由']]
    
    # 增加中文名稱欄位
    final_recommendations['股票名稱'] = final_recommendations['stock_name_en'].apply(
        lambda x: STOCK_NAMES_MAP.get(x, f'代碼{x}')
    )
    
    return final_recommendations

# -----------------------------------------------------------
# 步驟 3: Streamlit 網頁主體
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title='AI 短期波段選股系統', layout='wide')
    st.title('📈 AI 短期波段選股系統')
    st.markdown(f'**分析模型:** 基於 MACD, RSI, 成交量動能 | **價格限制:** ≤ {PRICE_LIMIT} 元.')
    
    # 設定日期範圍
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=120) 
    
    # 獲取並處理數據
    with st.spinner('🔄 正在獲取數據並運行分析模型...'):
        processed_data = get_and_prepare_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), SAMPLE_STOCKS)
    
    st.info(f'數據更新時間: {end_date.strftime('%Y-%m-%d')} (數據每 24 小時自動更新)')

    if processed_data.empty:
        st.error('❌ 無法獲取有效數據. 請檢查數據源或網路連接.')
        return

    # 運行分析模型
    final_recommendations = run_simple_momentum_model(processed_data)

    st.header('🏆 本日 Top 5 推薦清單')
    
    # 調整輸出順序和欄位名稱
    final_recommendations = final_recommendations[['stock_name_en', '股票名稱', 'Close', 'AI_Score', '推薦理由']]
    final_recommendations = final_recommendations.rename(columns={
        'stock_name_en': '股票代碼',
        'Close': '當日收盤價 (元)', 
        'AI_Score': '分析分數'
    })
    
    # 格式化輸出
    final_recommendations['當日收盤價 (元)'] = final_recommendations['當日收盤價 (元)'].apply(lambda x: f'{x:,.2f}')
    final_recommendations['分析分數'] = final_recommendations['分析分數'].apply(lambda x: f'{x:.2f}')

    # 使用 Streamlit 顯示表格
    st.dataframe(final_recommendations, use_container_width=True)

    # 使用明確的字符串和換行符來定義 Markdown, 避免三重引號衝突
    markdown_notes = (
        '---\n'
        '**備註說明:**\n'
        '- 分析分數為 [0.00, 0.70], 分數越高, 技術面訊號越強.\n'
        '- 推薦理由根據 MACD, RSI, 成交量等技術指標自動生成.'
    )
    st.markdown(markdown_notes)

if __name__ == '__main__':
    main()