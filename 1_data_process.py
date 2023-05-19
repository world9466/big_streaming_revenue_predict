import pandas as pd

# 讀取YT後台原始csv檔，刪掉第一筆total值

yt_data_2021 = pd.read_csv('origin_data/ytdata_2021.csv')
yt_data_2022 = pd.read_csv('origin_data/ytdata_2022.csv')
yt_data_2023 = pd.read_csv('origin_data/ytdata_2023.csv')

yt_data_2021 = yt_data_2021.tail(-1)
yt_data_2022 = yt_data_2022.tail(-1)
yt_data_2023 = yt_data_2023.tail(-1)

# 合併資料
yt_data = pd.concat([yt_data_2021,yt_data_2022,yt_data_2023],ignore_index = True)

# 按日期排列，重設索引
yt_data = yt_data.sort_values(['Date'],ascending=True)
yt_data = yt_data.reset_index(drop = True)

# 選取需要的欄位
yt_data_model = yt_data[[
    'Date',
    'YouTube Premium (USD)',
    'Transaction revenue (USD)',
    'Watch Page ads (USD)',
    'Estimated revenue (USD)',
    'Impressions',
    'Impressions click-through rate (%)'
    ]]

# 修改欄位名稱
yt_data_model = yt_data_model.rename(columns={
    'YouTube Premium (USD)':'Premium',
    'Transaction revenue (USD)':'Transaction',
    'Watch Page ads (USD)':'ads',
    'Estimated revenue (USD)':'total_revenue',
    'Impressions':'imps',
    'Impressions click-through rate (%)':'CTR'})

# 用總觀看數減去因為曝光點擊而來的次數，來評比大直播的非YT曝光收看成績
not_imp = yt_data['Views'] - round(yt_data_model['imps'] * yt_data_model['CTR']*0.01,0)
not_imp = {'not_imp':not_imp}
not_imp = pd.DataFrame(not_imp)

yt_data_model = yt_data_model.join(not_imp,rsuffix = 'repeated')

# 輸出模型訓練用的數據(5月份為預測用故刪除)
yt_data_model.to_excel('origin_data/yt_origin_data.xlsx',index=False)
