import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


################# 特徵預測(imps、CTR、not_imp) #################

# 讀取原始資料
data = pd.read_excel('origin_data/origin_data.xlsx')

pd_days = 169

# 把要預測的日期去掉
data_1 = data.head(-pd_days)

# 把預測目標提取出來做新的特徵
var_table = data_1[['imps','CTR','watch_time','not_imp']]

# 迭代要預測的天數，也可以更久只是會越來越不準
for num in range(pd_days):

    # 製作7日移動平均線，及7日內最大值當作特徵，重新命名欄位
    mavg_7d = var_table.rolling(window=7).mean()
    mavg_7d.columns=['imps_mavg','CTR_mavg','watch_time_mavg','not_imp_mavg']

    max_7d = var_table.rolling(window=7).max()
    max_7d.columns=['imps_max','CTR_max','watch_time_max','not_imp_max']
    
    # 合併特徵
    future_table = mavg_7d.join(max_7d,rsuffix = 'repeated')
    future_table = future_table.reset_index(drop = True)

    # 準備預測的日期特徵，隨著迭代推進要預測的日期
    data_2 = data.head(-pd_days+num).drop([
        'Date','Premium','Transaction','ads','total_revenue','imps','CTR','watch_time','not_imp','月底到月初'
        ],axis=1)

    # 把空值補為0，重設index避免join錯誤
    data_2.fillna(value = 0 ,inplace=True)
    data_2 = data_2.reset_index(drop = True)

    # 合併future_table，把前6筆沒有7日平均數的資料刪掉
    x = data_2.join(future_table,rsuffix = 'repeated').tail(-6)

    # 資料標準化
    scaler = StandardScaler()
    scaler.fit(x)
    x = pd.DataFrame(data=scaler.transform(x),columns=x.columns, index=x.index)


    # 載入訓練好的模型進行預測，只取最後一筆當作預測資料用
    with open('model/XGBOOST_future.pickle','rb') as f:
        xgbr = pickle.load(f)
        pred = xgbr.predict(x.tail(1))

    pred = pd.DataFrame(pred)
    pred.columns=['imps','CTR','watch_time','not_imp']

    var_table = pd.concat([var_table,pred],ignore_index = True)


################# 未來營收預測 #################

# 建立最終預測用特徵資料，把NaN取代為0
final_data = var_table.join(data.drop([
        'Date','Premium','Transaction','ads','total_revenue','imps','CTR','watch_time','not_imp'
        ],axis=1))
final_data.fillna(value = 0 ,inplace=True)


# 資料標準化，標準化完再轉成dataframe
scaler = StandardScaler()
scaler.fit(final_data)
final_data = pd.DataFrame(data=scaler.transform(final_data),columns=final_data.columns, index=final_data.index)


# 將特徵輸入到建立好的模型
with open('model/XGBOOST_muti.pickle','rb') as f:
    xgbr = pickle.load(f)
    final_pred = xgbr.predict(final_data)

final_pred = pd.DataFrame(final_pred)
final_pred.columns=['Premium','Transaction','ads']

# 將結果加總輸出dataframe
total_revenue_pred = {'pred_revenue':round(final_pred['Premium']+final_pred['Transaction']+final_pred['ads'],3)}
total_revenue_pred = pd.DataFrame(total_revenue_pred)

# 合併預測與原始收入
origin_data = pd.read_excel('origin_data/origin_data.xlsx')
compare_table = pd.DataFrame(origin_data[['Date','total_revenue']]).join(total_revenue_pred,rsuffix='_repeated',how = 'outer')

compare_table.to_excel('output/future_pred.xlsx',index = False)

