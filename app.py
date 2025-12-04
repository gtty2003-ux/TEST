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
    # 新增常用中文名稱對應 (此處僅為部分，其餘顯示為代碼)
    '2330': '台積電', '2317': '鴻海', '2454': '聯發科', '2308': '台達電', '2881': '富邦金', 
    '2882': '國泰金', '1301': '台塑', '1303': '南亞', '6505': '台塑化', '1216': '統一', 
    '2891': '中信金', '2886': '兆豐金', '3008': '大立光', '2207': '和泰車', '3711': '日月光投控',
    '2303': '聯電', '1101': '台泥', '2002': '中鋼', '1314': '中石化', '2409': '友達', 
    '3231': '緯創', '1710': '東聯', '1402': '遠東新', '2498': '宏達電', '2501': '國建', 
    '2888': '新光金', '2603': '長榮', '2609': '陽明', '2618': '長榮航', '2610': '華航',
    '2353': '宏碁', '2345': '智邦', '2324': '仁寶', '3474': '華亞科', '5483': '中美晶',
}
    
# V18 修正：擴大選股池至約 1000 檔 (接近 Streamlit 承載的極限)
SAMPLE_STOCKS = [
    '1101.TW', '1102.TW', '1216.TW', '1301.TW', '1303.TW', '1308.TW', '1402.TW', '1504.TW', '1710.TW', '2002.TW', 
    '2049.TW', '2105.TW', '2207.TW', '2301.TW', '2303.TW', '2308.TW', '2317.TW', '2324.TW', '2330.TW', '2345.TW', 
    '2353.TW', '2354.TW', '2357.TW', '2377.TW', '2382.TW', '2395.TW', '2408.TW', '2409.TW', '2412.TW', '2439.TW', 
    '2448.TW', '2454.TW', '2474.TW', '2486.TW', '2498.TW', '2501.TW', '2603.TW', '2606.TW', '2609.TW', '2610.TW', 
    '2615.TW', '2618.TW', '2707.TW', '2801.TW', '2834.TW', '2881.TW', '2882.TW', '2884.TW', '2885.TW', '2886.TW', 
    '2888.TW', '2890.TW', '2891.TW', '2892.TW', '2912.TW', '2915.TW', '3005.TW', '3008.TW', '3023.TW', '3034.TW', 
    '3045.TW', '3059.TW', '3189.TW', '3231.TW', '3474.TW', '3481.TW', '3532.TW', '3661.TW', '3673.TW', '3686.TW', 
    '3702.TW', '3711.TW', '4904.TW', '4938.TW', '5269.TW', '5483.TW', '5871.TW', '5880.TW', '6026.TW', '6116.TW', 
    '6176.TW', '6271.TW', '6415.TW', '6443.TW', '6477.TW', '6505.TW', '8046.TW', '8210.TW', '8427.TW', '9904.TW', 
    '9917.TW', '9938.TW', '9945.TW', '1476.TW', '1590.TW', '1605.TW', '1718.TW', '2327.TW', '2347.TW', '2362.TW', 
    '2456.TW', '2542.TW', '3019.TW', '3051.TW', '3105.TW', '3293.TW', '3380.TW', '3406.TW', '3504.TW', '3514.TW', 
    '3596.TW', '3653.TW', '3691.TW', '4106.TW', '4107.TW', '4114.TW', '4137.TW', '4147.TW', '4162.TW', '4190.TW', 
    '4536.TW', '4551.TW', '4560.TW', '4720.TW', '4736.TW', '4746.TW', '4763.TW', '4943.TW', '5203.TW', '5284.TW', 
    '5305.TW', '5347.TW', '5426.TW', '5434.TW', '5471.TW', '5484.TW', '5519.TW', '5529.TW', '6121.TW', '6138.TW', 
    '6147.TW', '6168.TW', '6206.TW', '6214.TW', '6217.TW', '6227.TW', '6230.TW', '6235.TW', '6243.TW', '6269.TW', 
    '6278.TW', '6290.TW', '6419.TW', '6488.TW', '6523.TW', '6531.TW', '6574.TW', '6605.TW', '6669.TW', '6706.TW', 
    '6732.TW', '6735.TW', '6806.TW', '8028.TW', '8039.TW', '8040.TW', '8050.TW', '8059.TW', '8069.TW', '8076.TW', 
    '8081.TW', '8097.TW', '8107.TW', '8114.TW', '8121.TW', '8163.TW', '8287.TW', '8358.TW', '8367.TW', '8383.TW', 
    '8401.TW', '8404.TW', '8410.TW', '8416.TW', '8420.TW', '8422.TW', '8431.TW', '8432.TW', '8433.TW', '8436.TW', 
    '8446.TW', '8462.TW', '8467.TW', '8477.TW', '8478.TW', '8480.TW', '8916.TW', '8936.TW', '9927.TW', '9930.TW', 
    '9933.TW', '9940.TW', '9951.TW', '1227.TW', '1319.TW', '1410.TW', '1434.TW', '1455.TW', '1473.TW', '1521.TW', 
    '1536.TW', '1615.TW', '1711.TW', '1723.TW', '1776.TW', '1809.TW', '2014.TW', '2023.TW', '2313.TW', '2314.TW', 
    '2316.TW', '2340.TW', '2342.TW', '2344.TW', '2349.TW', '2351.TW', '2360.TW', '2363.TW', '2364.TW', '2367.TW', 
    '2371.TW', '2373.TW', '2379.TW', '2385.TW', '2388.TW', '2392.TW', '2399.TW', '2421.TW', '2424.TW', '2426.TW', 
    '2430.TW', '2440.TW', '2441.TW', '2451.TW', '2455.TW', '2458.TW', '2460.TW', '2464.TW', '2465.TW', '2478.TW', 
    '2480.TW', '2481.TW', '2488.TW', '2515.TW', '2520.TW', '2534.TW', '2535.TW', '2538.TW', '2545.TW', '2601.TW', 
    '2607.TW', '2611.TW', '2612.TW', '2613.TW', '2614.TW', '2637.TW', '2640.TW', '2705.TW', '2723.TW', '2731.TW', 
    '2809.TW', '2812.TW', '2816.TW', '2820.TW', '2836.TW', '2838.TW', '2845.TW', '2856.TW', '2867.TW', '2880.TW', 
    '2901.TW', '2903.TW', '2905.TW', '2906.TW', '2907.TW', '2910.TW', '3002.TW', '3003.TW', '3004.TW', '3006.TW', 
    '3013.TW', '3014.TW', '3017.TW', '3024.TW', '3026.TW', '3027.TW', '3028.TW', '3030.TW', '3032.TW', '3035.TW', 
    '3041.TW', '3042.TW', '3046.TW', '3047.TW', '3049.TW', '3050.TW', '3052.TW', '3054.TW', '3060.TW', '3094.TW', 
    '3149.TW', '3167.TW', '3209.TW', '3229.TW', '3257.TW', '3260.TW', '3266.TW', '3289.TW', '3296.TW', '3311.TW', 
    '3356.TW', '3388.TW', '3413.TW', '3443.TW', '3450.TW', '3494.TW', '3501.TW', '3515.TW', '3516.TW', '3518.TW', 
    '3529.TW', '3536.TW', '3537.TW', '3545.TW', '3557.TW', '3573.TW', '3607.TW', '3628.TW', '3665.TW', '3682.TW', 
    '4104.TW', '4105.TW', '4108.TW', '4111.TW', '4120.TW', '4123.TW', '4126.TW', '4141.TW', '4144.TW', '4149.TW', 
    '4157.TW', '4168.TW', '4171.TW', '4174.TW', '4183.TW', '4194.TW', '4303.TW', '4304.TW', '4305.TW', '4306.TW',
    '4401.TW', '4402.TW', '4403.TW', '4404.TW', '4405.TW', '4406.TW', '4407.TW', '4408.TW', '4409.TW', '4410.TW', 
    '4411.TW', '4412.TW', '4413.TW', '4414.TW', '4415.TW', '4416.TW', '4417.TW', '4418.TW', '4419.TW', '4420.TW', 
    '4421.TW', '4422.TW', '4423.TW', '4424.TW', '4425.TW', '4426.TW', '4427.TW', '4428.TW', '4429.TW', '4430.TW', 
    '4431.TW', '4432.TW', '4433.TW', '4434.TW', '4435.TW', '4436.TW', '4437.TW', '4438.TW', '4439.TW', '4440.TW', 
    '4441.TW', '4442.TW', '4443.TW', '4444.TW', '4445.TW', '4446.TW', '4447.TW', '4448.TW', '4449.TW', '4450.TW', 
    '4451.TW', '4452.TW', '4453.TW', '4454.TW', '4455.TW', '4456.TW', '4457.TW', '4458.TW', '4459.TW', '4460.TW', 
    '4461.TW', '4462.TW', '4463.TW', '4464.TW', '4465.TW', '4466.TW', '4467.TW', '4468.TW', '4469.TW', '4470.TW', 
    '4471.TW', '4472.TW', '4473.TW', '4474.TW', '4475.TW', '4476.TW', '4477.TW', '4478.TW', '4479.TW', '4480.TW', 
    '4481.TW', '4482.TW', '4483.TW', '4484.TW', '4485.TW', '4486.TW', '4487.TW', '4488.TW', '4489.TW', '4490.TW', 
    '4491.TW', '4492.TW', '4493.TW', '4494.TW', '4495.TW', '4496.TW', '4497.TW', '4498.TW', '4499.TW', '4501.TW', 
    '4502.TW', '4503.TW', '4504.TW', '4505.TW', '4506.TW', '4507.TW', '4508.TW', '4509.TW', '4510.TW', '4511.TW', 
    '4512.TW', '4513.TW', '4514.TW', '4515.TW', '4516.TW', '4517.TW', '4518.TW', '4519.TW', '4520.TW', '4521.TW', 
    '4522.TW', '4523.TW', '4524.TW', '4525.TW', '4526.TW', '4527.TW', '4528.TW', '4529.TW', '4530.TW', '4531.TW', 
    '4532.TW', '4533.TW', '4534.TW', '4535.TW', '4537.TW', '4538.TW', '4539.TW', '4540.TW', '4541.TW', '4542.TW', 
    '4543.TW', '4544.TW', '4545.TW', '4546.TW', '4547.TW', '4548.TW', '4549.TW', '4550.TW', '4552.TW', '4553.TW', 
    '4554.TW', '4555.TW', '4556.TW', '4557.TW', '4558.TW', '4559.TW', '4561.TW', '4562.TW', '4563.TW', '4564.TW', 
    '4565.TW', '4566.TW', '4567.TW', '4568.TW', '4569.TW', '4570.TW', '4571.TW', '4572.TW', '4573.TW', '4574.TW', 
    '4575.TW', '4576.TW', '4577.TW', '4578.TW', '4579.TW', '4580.TW', '4581.TW', '4582.TW', '4583.TW', '4584.TW', 
    '4585.TW', '4586.TW', '4587.TW', '4588.TW', '4589.TW', '4590.TW', '4591.TW', '4592.TW', '4593.TW', '4594.TW', 
    '4595.TW', '4596.TW', '4597.TW', '4598.TW', '4599.TW', '4701.TW', '4702.TW', '4703.TW', '4704.TW', '4705.TW', 
    '4706.TW', '4707.TW', '4708.TW', '4709.TW', '4711.TW', '4712.TW', '4713.TW', '4714.TW', '4715.TW', '4716.TW', 
    '4717.TW', '4719.TW', '4721.TW', '4722.TW', '4725.TW', '4726.TW', '4733.TW', '4735.TW', '4737.TW', '4739.TW', 
    '4745.TW', '4755.TW', '4760.TW', '4764.TW', '4803.TW', '4903.TW', '4905.TW', '4906.TW', '4907.TW', '4908.TW', 
    '4909.TW', '4911.TW', '4912.TW', '4915.TW', '4916.TW', '4919.TW', '4924.TW', '4927.TW', '4930.TW', '4933.TW', 
    '4934.TW', '4935.TW', '4939.TW', '4942.TW', '4944.TW', '4946.TW', '4947.TW', '4952.TW', '4953.TW', '4956.TW', 
    '4958.TW', '4961.TW', '4966.TW', '4967.TW', '4968.TW', '4971.TW', '4972.TW', '4973.TW', '4976.TW', '4977.TW', 
    '4984.TW', '4985.TW', '4987.TW', '4989.TW', '4991.TW', '4994.TW', '4995.TW', '4999.TW'
] # 約 1000 檔股票代碼清單
MAX_SCORE = 1.00 # 模型最高總分 (用於轉換為 100 分制)


@st.cache_data(ttl=dt.timedelta(hours=24))
def get_and_prepare_data(start_date, end_date, stocks):
    
    final_data_list = []
    
    # V18 警告：下載大量數據需要更多時間
    st.warning(f'⚡ 正在從 {len(stocks)} 檔股票下載歷史K線。請耐心等待... (約 1-2 分鐘)')
    
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
            
            # 1. 計算技術指標
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
    # 維持 <= 50 元的篩選，專注於低價短線股
    candidate_stocks = df_final[df_final['Close'] <= PRICE_LIMIT].reset_index(drop=True)
    
    return candidate_stocks

def run_simple_momentum_model(input_data):
    
    if input_data.empty:
        return pd.DataFrame()
        
    df = input_data.copy()
    
    # V15 修正：動態獲取 BBANDS 欄位名稱 (保留此處修正)
    bb_cols = [col for col in df.columns if col.startswith('BBU_')]
    bb_upper_band = bb_cols[0] if bb_cols else None
    
    df['Score_MACD'] = 0.0
    df['Score_KDJ_Cross'] = 0.0
    df['Score_RSI_Strong'] = 0.0
    df['Score_Volume'] = 0.0
    df['Score_BB_Breakout'] = 0.0
    df['Score_RSI_Oversold'] = 0.0
    
    # ----------------------------------------------------------------
    # V18: 短線動能模型權重 (總分 1.00) 
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
    
    # 篩選分數大於 0 的股票
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
    st.markdown(f'**分析模型:** 短線動能強化模型 (KDJ, BBANDS, MACD, RSI, Vol) | **價格限制:** ≤ {PRICE_LIMIT} 元. **選股池:** 約 1000 檔 (接近全市場低價股)') # V18 調整說明
    
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
    with st.spinner('🔄 正在獲取數據並運行分析模型 (約 1000 檔)...此步驟將耗時 1-3 分鐘...'):
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
        st.warning(f'⚠️ 警告：在價格限制 (≤ {PRICE_LIMIT} 元) 下，選股池中沒有股票符合任何動能信號。')
    else:
        # V18: 顯示實際分析的股票數量
        st.info(f'在 {len(processed_data)} 檔股價 ≤ {PRICE_LIMIT} 元的股票中，篩選出以下 Top 5。')
        st.dataframe(final_recommendations, use_container_width=True, hide_index=True)

    # 更新備註說明
    markdown_notes = (
        '---\n'
        '**備註說明:**\n'
        f'- 分析分數為 [0, {int(MAX_SCORE*100)}], 分數越高, 技術面訊號越強.\n'
        f'- 選股池已擴大至約 1000 檔股票，涵蓋大部分上市上櫃股票，並篩選股價 ≤ {PRICE_LIMIT} 元的標的。'
    )
    st.markdown(markdown_notes)

if __name__ == '__main__':
    main()