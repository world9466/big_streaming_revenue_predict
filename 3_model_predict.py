import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# 讀取處理好的數據，空位都補成0
data = pd.read_excel('origin_data/origin_data.xlsx').head(-15)
data.fillna(value = 0 ,inplace=True)

# 刪除不必要的欄位
x = data.drop(['Date','Premium','Transaction','ads','total_revenue'],axis=1)
# 資料標準化
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# 載入訓練好的模型進行預測
with open('model/XGBOOST_muti.pickle','rb') as f:
    xgbr = pickle.load(f)
    pred = xgbr.predict(x)

# 轉為dataframe方便運算
pred = pd.DataFrame(pred)
pred.columns=['Premium','Transaction','ads']

# 將結果加總輸出dataframe
total_revenue_pred = {'pred_revenue':(pred['Premium']+pred['Transaction']+pred['ads'])}
total_revenue_pred = pd.DataFrame(total_revenue_pred)

# 合併預測與原始收入
compare_table = pd.DataFrame(data[['Date','total_revenue']]).join(total_revenue_pred,rsuffix='_repeated')


compare_table.to_excel('output/history_pred.xlsx',index = False)