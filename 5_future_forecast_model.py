import os,time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,make_scorer, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from tqdm import tqdm

# 讀取原始資料，把要預測的日期去掉
data_1 = pd.read_excel('origin_data/origin_data.xlsx').head(-169)

# 把預測目標提取出來做新的特徵
var_table = data_1[['imps','CTR','watch_time','not_imp']]

# 製作7日移動平均線&7日內最大值當作特徵，重新命名欄位
mavg_7d = var_table.rolling(window=7).mean()
mavg_7d.columns=['imps_mavg','CTR_mavg','watch_time_mavg','not_imp_mavg']

max_7d = var_table.rolling(window=7).max()
max_7d.columns=['imps_max','CTR_max','watch_time_max','not_imp_max']


# 取前7日平均及前7日內最大值為新的特徵，做法是把最後一天刪除，第一天補一筆空白資料，避免合併時錯誤(筆數不同)
mavg_7d = mavg_7d.head(-1)
max_7d = max_7d.head(-1)

# 修改7日平均值
new_table = {'imps_mavg':[np.NAN],'CTR_mavg':[np.NAN],'watch_time_mavg':[np.NAN],'not_imp_mavg':[np.NAN]}
new_table = pd.DataFrame(new_table)
mavg_7d = pd.concat([new_table,mavg_7d],ignore_index = True)

# 修改7日內最大值
new_table = {'imps_max':[np.NAN],'CTR_max':[np.NAN],'watch_time_max':[np.NAN],'not_imp_max':[np.NAN]}
new_table = pd.DataFrame(new_table)
max_7d = pd.concat([new_table,max_7d],ignore_index = True)



# 與原特徵表格合併
data_2 = data_1.drop(['Date','Premium','Transaction','ads','total_revenue','月底到月初'],axis=1)

train_table = data_2.join(mavg_7d,rsuffix = 'repeated')
train_table = train_table.join(max_7d,rsuffix = 'repeated')

# 去除前7筆有缺值的筆數，把空值補為0
train_table = train_table.tail(-7)
train_table.fillna(value = 0 ,inplace=True)



################# 模型訓練與測試 #################

# 分割資料
y = train_table[['imps','CTR','watch_time','not_imp']]   # 採用多輸出預測
x = train_table.drop(['imps','CTR','watch_time','not_imp'],axis=1)


# 資料標準化
scaler = StandardScaler()
scaler.fit(x)
x = pd.DataFrame(data=scaler.transform(x),columns=x.columns, index=x.index)

# 決定訓練與測試的比重
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25, random_state=30)

# 建立 XGBOOST 模型
xgbr = XGBRegressor()

# 模型參數前面一定要加上 estimator__
param_grid = {
    'estimator__learning_rate':np.arange(0.05,0.11,0.01), # 學習率0.05~0.10，遞增值為0.01
    'estimator__n_estimators':range(30,100),              # 決策樹(子模型)數量，預設為100
    'estimator__max_depth':range(4,9),                    # 決策樹深度，預設為3
    'estimator__min_child_weight':[3],
    'estimator__gamma':[0],
    'estimator__subsample':[0.8],
    'estimator__colsample_bytree':[0.8],
    'estimator__objective':['reg:squarederror'],
    'estimator__reg_lambda':[0],
    'estimator__reg_alpha':[0],
    'estimator__nthread':[4],
    'estimator__scale_pos_weight':[1],
    'estimator__seed':[168]
    }

# 多輸出回歸，採用XGBOOST
mogr = MultiOutputRegressor(xgbr)

# 修改GridSearchCV的評估指標
scorer = make_scorer(r2_score)

# 將模型套入grid_search(網格搜尋)，交叉驗證的k值(cv)設為5，評估指標使用決定係數r2
#grid_search = GridSearchCV(estimator=mogr , param_grid=param_grid,cv=5 , scoring=scorer)

# 訓練時間太久可以試試RandomizeSearchCV()，n_iter可以指定隨機比數
grid_search = RandomizedSearchCV(estimator=mogr , param_distributions=param_grid,cv=5 , scoring=scorer , n_iter=20)


# 獲取要搜索的參數組合的總數
total_combinations = len(param_grid)


# 使用訓練組開始訓練
print('開始搜尋最佳參數值...(如果時間太久可以試著啟用RandomizeSearchCV())')
start = time.time()

################################################
# 使用tqdm來創建進度條
for i, params in enumerate(tqdm(param_grid, desc="Grid Search Progress")):
    # 開始訓練
    grid_search.fit(X_train, y_train)

    # 輸出當前參數組合的結果
    print(f"Params: {params}, Best score (R^2): {grid_search.best_score_}")

    # 更新進度條
    tqdm.write(f"Progress: {i+1}/{total_combinations} ({((i+1)/total_combinations)*100:.2f}%)")

################################################

end = time.time()
print('資料計算共耗時',round(end - start,2),'秒')
# 列出最好的參數組合及分數
print("Best parameters found: ", grid_search.best_params_)
print("Best score(決定係數): ", grid_search.best_score_)

# 取出最佳參數，把字典的key前綴estimator__刪除，不然無法代入xgboost
best_params = {key.replace("estimator__", ""): value for key, value in grid_search.best_params_.items()}

# 代入xgboost模型進行多輸出訓練
xgbr = XGBRegressor(**best_params)
mogr = MultiOutputRegressor(xgbr).fit(X_train, y_train)


# 訓練組
# 多輸出訓練，轉成dataframe，更改欄位名稱
y_pred_Mu_train = mogr.predict(X_train)
y_pred_Mu_train = pd.DataFrame(y_pred_Mu_train)
y_pred_Mu_train.columns=['imps','CTR','watch_time','not_imp']


# 測試組
y_pred_Mu_test = mogr.predict(X_test)
y_pred_Mu_test = pd.DataFrame(y_pred_Mu_test)
y_pred_Mu_test.columns=['imps','CTR','watch_time','not_imp']

# 迭代每個預測值與實際值觀察準確率
for target in ['imps','CTR','watch_time','not_imp']:

    # 訓練組結果
    print()
    print('XGBOOST模型訓練效果(預測目標：{})'.format(target))
    MSE = mean_squared_error(y_train[target], y_pred_Mu_train[target])
    print('MSE=',MSE)
    RMSE =np.sqrt(MSE)
    print('RMSE=',RMSE)
    MAE= mean_absolute_error(y_train[target], y_pred_Mu_train[target])
    print('MAE=',MAE)
    R2=1-MSE/np.var(y_train[target])
    print("R2:", R2)


    # 測試組結果
    print()
    print('XGBOOST模型測試效果(預測目標：{})'.format(target))
    MSE = mean_squared_error(y_test[target], y_pred_Mu_test[target])
    print('MSE=',MSE)
    RMSE =np.sqrt(MSE)
    print('RMSE=',RMSE)
    MAE= mean_absolute_error(y_test[target], y_pred_Mu_test[target])
    print('MAE=',MAE)
    R2=1-MSE/np.var(y_test[target])
    print("R2:", R2)



    ##### XGBOOST門檻測試 #####
    # 門檻測試，(實際值-測試值)/實際值，每筆去計算百分比後有通過門檻的再與總筆數做百分比

    y_list = []
    for value in y_test[target]:
        if value != 0:            # 若預測值為 0，則取代為 1 避免出現分母出現 0
            y_list.append(value)
        else:
            y_list.append(1)


    #y_list = np.array(y_list) # 若沒有把計算結果轉為dataframe，要啟用此列

    # abs 把每個計算結果絕對值化
    errvalue = abs((y_list-y_pred_Mu_test[target])/y_list)

    # 門檻值
    threshold=0.1
    countXG=0

    for i in range(len(errvalue)):
        if errvalue[i]<threshold:        
            countXG=countXG+1

    # 計算通過率
    pass_rate = round(countXG/len(errvalue),2)
    print('XGBOOST模型在門檻正負{}%內，通過率為{}%'.format(int(threshold*100),int(pass_rate*100)))


    # 平均誤差百分比
    merror_pct = errvalue.sum()/len(errvalue)
    print('預測值平均誤差百分比為正負{}%'.format(int(round(merror_pct,3)*1000)/10))
    print()
    print('----------------------------------')


############### 輸出儲存預測模型 ###############

import pickle

# 把訓練好的模型儲存為pickle檔案 - XGBOOST
with open('model/XGBOOST_future.pickle','wb') as f:
   pickle.dump(mogr,f)



