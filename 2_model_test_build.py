import os,time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,make_scorer, r2_score
from math import sqrt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# 讀取處理好(去除不合理的項目)的資料，空位都補成0，processed_data已經把要預測的5月份去掉
data = pd.read_excel('processed_data/processed_data.xlsx')
data.fillna(value = 0 ,inplace=True)


y = data[['Premium','Transaction','ads']]   # 採用多輸出預測
x = data.drop(['Date','Premium','Transaction','ads','total_revenue'],axis=1)


# 資料標準化
scaler = StandardScaler()
scaler.fit(x)
x = pd.DataFrame(data=scaler.transform(x),columns=x.columns, index=x.index)


'''
# 特定範圍標準化，預設0~1
minmaxscaler=MinMaxScaler()
x=minmaxscaler.fit_transform(x)
print(x)
'''


# 決定訓練與測試的比重
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25, random_state=30)



############### XGBOOST ###############
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
grid_search = GridSearchCV(estimator=mogr , param_grid=param_grid,cv=5 , scoring=scorer)

# 訓練時間太久可以試試RandomizeSearchCV()，n_iter可以指定隨機比數
#grid_search = RandomizedSearchCV(estimator=mogr , param_grid=param_grid,cv=5 , scoring=scorer , n_iter=10)

# 使用訓練組開始訓練
print('開始搜尋最佳參數值...(如果時間太久可以試著啟用RandomizeSearchCV()')
start = time.time()

grid_search.fit(X_train, y_train)

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


# 多輸出訓練，轉成dataframe，更改欄位名稱
y_pred_Mu_train = mogr.predict(X_train)
y_pred_Mu_train = pd.DataFrame(y_pred_Mu_train)
y_pred_Mu_train.columns=['Premium','Transaction','ads']

# 測試組
y_pred_Mu_test = mogr.predict(X_test)
y_pred_Mu_test = pd.DataFrame(y_pred_Mu_test)
y_pred_Mu_test.columns=['Premium','Transaction','ads']


# 迭代每個預測值與實際值觀察準確率
for target in ['Premium','Transaction','ads']:

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
        y_list.append(value)

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


# 計算總收益誤差

# 先加原始及預測的總收益
total_y_test = y_test['Premium']+y_test['Transaction']+y_test['ads']
total_y_pred = y_pred_Mu_test['Premium']+y_pred_Mu_test['Transaction']+y_pred_Mu_test['ads']

# 轉換為陣列進行運算
total_y_test = np.array(total_y_test)
total_y_pred = np.array(total_y_pred)

# 計算百分比
total_merror = abs((total_y_test-total_y_pred)/total_y_test)
total_merror_pct = total_merror.sum()/len(total_merror)


print('\n','總收益平均誤差百分比為正負{}%'.format(int(round(total_merror_pct,3)*1000)/10),'\n')

print('----------------------------------')


############### 輸出儲存預測模型 ###############

import pickle


# 把訓練好的模型儲存為pickle檔案 - XGBOOST
with open('model/XGBOOST_muti.pickle','wb') as f:
   pickle.dump(mogr,f)

