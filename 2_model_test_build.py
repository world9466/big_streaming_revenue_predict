import os,time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,make_scorer, r2_score
from math import sqrt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


# 可以建立processed_data將不合理的資料刪除，或是直接取用origin_data.xlsx，空位都補成0
data_path = 'processed_data/processed_data.xlsx'
if os.path.isfile(data_path):
    data = pd.read_excel('processed_data/processed_data.xlsx')
else:
    data = pd.read_excel('origin_data/origin_data.xlsx')

data.fillna(value = 0 ,inplace=True)


############### 特徵相關性分析 ###############

# 選擇需要進行相關性分析的目標變量和其他特徵
target_variable = data[['Premium','Transaction','ads','imps','not_imp','CTR','watch_time']]

# 計算相關性矩陣
correlation_matrix = target_variable.corr()

# 將相關性矩陣轉換為DataFrame
correlation_table = pd.DataFrame(correlation_matrix, columns=target_variable.columns, index=target_variable.columns)

# 去除不必要的觀察選項
correlation_table = correlation_table.head(3).drop(['Premium','Transaction','ads'],axis=1)

# 繪製熱圖
plt.figure(figsize=(12, 9)) 
sns.heatmap(correlation_table, vmin=-1,vmax=1, annot=True, cmap='coolwarm', annot_kws={'fontsize': 30})
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']
plt.title('皮爾森積差相關分析',fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('output/heatmap.png')



############### 決定特徵變數與預測變數 ###############
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

# 模型參數前面一定要加上 estimator__，已經找過最佳參數，所以範圍已經限縮過
param_grid = {
    'estimator__learning_rate':np.arange(0.05,0.11,0.01), # 學習率0.05~0.10，遞增值為0.01
    'estimator__n_estimators':range(30,80),              # 決策樹(子模型)數量，預設為100
    'estimator__max_depth':[6],                    # 決策樹深度，預設為3
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
#grid_search = GridSearchCV(estimator=mogr , param_grid=param_grid, cv=5 , scoring=scorer)

# 訓練時間太久可以試試RandomizeSearchCV()，n_iter可以指定隨機筆數
grid_search = RandomizedSearchCV(estimator=mogr , param_distributions=param_grid,cv=5 , scoring=scorer , n_iter=20)

# 使用訓練組開始訓練
print('開始搜尋最佳參數值...如果時間太久可以試著啟用RandomizeSearchCV()')
start = time.time()

grid_search.fit(X_train, y_train)

end = time.time()
print('資料計算共耗時',round(end - start,2),'秒')
# 列出最好的參數組合及分數
print("Best parameters found: ", grid_search.best_params_)
print("Best score(決定係數): ", grid_search.best_score_)

# 寫入模型報告
path = '主模型建立及預測參數報告.txt'
f = open(path, 'w')
f.write('Best parameters found:')
f.write(str(grid_search.best_params_))
f.write('\nBest score(決定係數):')
f.write(str(grid_search.best_score_))
f.close

# 取出最佳參數，把字典的key前綴estimator__刪除，不然無法代入xgboost
best_params = {key.replace("estimator__", ""): value for key, value in grid_search.best_params_.items()}


# 特徵重要性分析
for pv in ['Premium','Transaction','ads']:
    xgbr = XGBRegressor(**best_params).fit(X_train, y_train[pv])
    feature_importance = xgbr.feature_importances_
    feature_importance_table = pd.DataFrame({'feature': x.columns, 'importance': feature_importance})

    feature_importance_table = feature_importance_table.sort_values('importance', ascending=False)
    feature_importance_table = feature_importance_table.reset_index(drop = True)

    print('\n=====預測 {} 時，特徵之間相對重要性=====\n\n'.format(pv),feature_importance_table)

    # 寫入報告
    f = open(path, 'a')
    f.write('\n=====預測 {} 時，特徵之間相對重要性=====\n'.format(pv))
    f.write(feature_importance_table.to_string())
    f.close


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

    f = open(path, 'a')

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
    print('R2:', R2)

    f.write('\n\nXGBOOST模型訓練效果(預測目標：{})'.format(target))
    f.write('\nMSE={}'.format(MSE))
    f.write('\nRMSE={}'.format(RMSE))
    f.write('\nR2:{}'.format(R2))

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

    f.write('\n\nXGBOOST模型測試效果(預測目標：{})'.format(target))
    f.write('\nMSE={}'.format(MSE))
    f.write('\nRMSE={}'.format(RMSE))
    f.write('\nR2:{}'.format(R2))


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

    f.write('\n預測值平均誤差百分比為正負{}%'.format(int(round(merror_pct,3)*1000)/10))
    f.close


# 計算總收益誤差

# 先加原始及預測的總收益，為避免分母為0出現無限大的數字，取代0為1
total_y_test = y_test['Premium']+y_test['Transaction']+y_test['ads']
total_y_test[total_y_test == 0] = 1
total_y_pred = y_pred_Mu_test['Premium']+y_pred_Mu_test['Transaction']+y_pred_Mu_test['ads']

# 轉換為陣列進行運算
total_y_test = np.array(total_y_test)
total_y_pred = np.array(total_y_pred)

# 計算百分比
total_merror = abs((total_y_test-total_y_pred)/total_y_test)
total_merror_pct = total_merror.sum()/len(total_merror)


print('\n總收益平均誤差百分比為正負{}%\n'.format(int(round(total_merror_pct,3)*1000)/10))

print('----------------------------------')

f = open(path, 'a')
f.write('\n\n總收益平均誤差百分比為正負{}%'.format(int(round(total_merror_pct,3)*1000)/10))
f.close

############### 輸出儲存預測模型 ###############

import pickle


# 把訓練好的模型儲存為pickle檔案 - XGBOOST
with open('model/XGBOOST_muti.pickle','wb') as f:
   pickle.dump(mogr,f)

