import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


# 讀取處理好(去除不合理的項目)的資料，空位都補成0，processed_data已經把要預測的5月份去掉
data = pd.read_excel('processed_data/processed_data.xlsx')
data.fillna(value = 0 ,inplace=True)

# 分割資料
#y = data['Premium'].round(0)   # 單一預測

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


'''
############### DNN ###############
# 建立 DNN 模型
nn=MLPRegressor(hidden_layer_sizes=(3,1),  #預設值1層,數目100
         activation='relu',         #{'identity'無，'logistic'為sigmoid，'tanh'，'relu'}，預設='relu'
         solver='sgd',             #{'lbfgs'牛頓，'sgd'隨機梯度，'adam'隨機梯度優化}，預設='adam'
         batch_size='auto',         #'auto'為min(200～樣本數)
         learning_rate='constant',  #constant固定'，'invscaling隨著時間遞減'，'adaptive誤差減少時不改變'}，預設 ='constant' 
         learning_rate_init=0.001,  #優化器為sgd、adam使用
         power_t=0.5,               #優化器為sgd，則可以設定強化學習率   
         max_iter=2000,             #訓練次數
         shuffle=True,              #隨機設定權重
         random_state=0,            #隨機的基礎值
         momentum=0.9
         )                          #默認 0.9,動量梯度下降更新，設置的範圍應該0.0-1.0. 只有solver=’sgd’時使用

# 帶入模型
nn.fit(X_train, y_train)
y_pred_NN_train=nn.predict(X_train)

print()
print('DNN模型訓練效果')
# 計算訓練組的均方誤差(真實值與預估值的平均差異)
MSE = mean_squared_error(y_train, y_pred_NN_train)
print('MSE=',MSE)

# 計算訓練組的標準差(均方差)，把均方誤差開根號
RMSE =np.sqrt(MSE)
print('RMSE=',RMSE)

# 平均絕對誤差，與標準差類似，但直接取差值的絕對值，不用平方相加再開根號，對於極值的反應不會那麼強烈
# 如果數據極值多可以參考這個值
MAE= mean_absolute_error(y_train, y_pred_NN_train)
print('MAE=',MAE)

# R2：決定係數，代表這個模型的解釋力，越接近1越好，負值代表預估值誤差比直接用平均值誤差還要大(爛)
R2=1-MSE/np.var(y_train)
print("R2:", R2)

print()
print('DNN模型測試效果')
# 測試組
y_pred_NN_test=nn.predict(X_test)

MSE = mean_squared_error(y_test, y_pred_NN_test)
print('MSE=',MSE)
RMSE =np.sqrt(MSE)
print('RMSE=',RMSE)
MAE= mean_absolute_error(y_test, y_pred_NN_test)
print('MAE=',MAE)

R2=1-MSE/np.var(y_test)
print("R2:", R2)


##### DNN門檻測試 #####
# 門檻測試，(實際值-測試值)/實際值，每筆去計算百分比後有通過門檻的再與總筆數做百分比
# 看正負多少%能通過95%門檻

# 取出實際的y為list
y_list = []
for value in y_test:
    y_list.append(value)

# y_pred_NN_test 為 numpy.ndarray 格式，y_list也要轉換
y_list = np.array(y_list)

# 每筆計算後變為一個陣列
errvalue=abs((y_list-y_pred_NN_test)/y_list)

# 門檻值
threshold=0.2
countNN=0

# 迭代全部資料筆數，如果符合門檻，就加1
for i in range(len(errvalue)):
  if errvalue[i]<threshold:        
    countNN=countNN+1

# 計算通過率，通過數除以總筆數
pass_rate = round(countNN/len(errvalue),2)
print('DNN模型在門檻正負{}%內，通過率為{}%'.format(int(threshold*100),int(pass_rate*100)))

print('----------------------------------')



############### SVM & Non-linear SVMs ###############
# 建立 線性支持向量機 模型

linear_svm = LinearSVC()
linear_svm.fit(X_train, y_train)

y_pred_linear_svm_train = linear_svm.predict(X_train)

print()
print('線性SVM模型訓練效果')
MSE = mean_squared_error(y_train, y_pred_linear_svm_train)
print('MSE=',MSE)
RMSE =np.sqrt(MSE)
print('RMSE=',RMSE)
MAE= mean_absolute_error(y_train, y_pred_linear_svm_train)
print('MAE=',MAE)

R2=1-MSE/np.var(y_train)
print("R2:", R2)


print()
print('線性SVM模型測試效果')
# 測試組
svmtest_pred=linear_svm.predict(X_test)

MSE = mean_squared_error(y_test, svmtest_pred)
print('MSE=',MSE)
RMSE =np.sqrt(MSE)
print('RMSE=',RMSE)
MAE= mean_absolute_error(y_test, svmtest_pred)
print('MAE=',MAE)

R2=1-MSE/np.var(y_test)
print("R2:", R2)


##### SVM門檻測試 #####
# 門檻測試，(實際值-測試值)/實際值，每筆去計算百分比後有通過門檻的再與總筆數做百分比
# 看正負多少%能通過95%門檻

y_list = []
for value in y_test:
    y_list.append(value)

y_list = np.array(y_list)

errvalue=abs((y_list-svmtest_pred)/y_list)

# 門檻值
threshold=0.2
countSVM=0

for i in range(len(errvalue)):
  if errvalue[i]<threshold:        
    countSVM=countSVM+1

# 計算通過率
pass_rate = round(countSVM/len(errvalue),2)
print('線性SVM模型在門檻正負{}%內，通過率為{}%'.format(int(threshold*100),int(pass_rate*100)))

print('----------------------------------')



# 建立 非線性支持向量機 模型
svm = SVC(
    C=1.0,                   #目標函數的懲罰係數，預設1.0
    cache_size=500,          #制定訓練所需要的內存，緩存大小會影響訓練速度，預設200
    class_weight=None,       #指定樣本各類別的的權重
    coef0=0.0,               #核函數中的獨立項，'RBF' and 'sigmoid'有效
    decision_function_shape='ovo',    #二元分類(ovr)或多次二元分類(ovo)，ovo較精確但速度較慢，預設ovr
    degree=3,                #如果kernel使用多項式核函數, degree決定了多項式的最高次冪
    gamma='auto',            #核函數的係數
    kernel='rbf',            #選擇有rbf(高斯核)(預設), linear(線性核函數), poly(多項式核函數), Sigmoid(sigmoid核函數)
    max_iter=-1,             #最大疊代次數，預設為1
    probability=False,       #可能性估計
    random_state=None,
    shrinking=True,
    tol=0.001,               #svm結束標準的精度
    verbose=False)

svm.fit(X_train, y_train)

y_pred_svm_train = svm.predict(X_train)

print()
print('非線性SVM模型訓練效果')
MSE = mean_squared_error(y_train, y_pred_svm_train)
print('MSE=',MSE)
RMSE =np.sqrt(MSE)
print('RMSE=',RMSE)
MAE= mean_absolute_error(y_train, y_pred_svm_train)
print('MAE=',MAE)

R2=1-MSE/np.var(y_train)
print("R2:", R2)


print()
print('非線性SVM模型測試效果')
# 測試組
nsvmtest_pred=svm.predict(X_test)

MSE = mean_squared_error(y_test, nsvmtest_pred)
print('MSE=',MSE)
RMSE =np.sqrt(MSE)
print('RMSE=',RMSE)
MAE= mean_absolute_error(y_test, nsvmtest_pred)
print('MAE=',MAE)

R2=1-MSE/np.var(y_test)
print("R2:", R2)


##### 非線性SVM門檻測試 #####
# 門檻測試，(實際值-測試值)/實際值，每筆去計算百分比後有通過門檻的再與總筆數做百分比
# 看正負多少%能通過95%門檻

y_list = []
for value in y_test:
    y_list.append(value)

y_list = np.array(y_list)

errvalue=abs((y_list-nsvmtest_pred)/y_list)

# 門檻值
threshold=0.2
countNSVM=0

for i in range(len(errvalue)):
  if errvalue[i]<threshold:        
    countNSVM=countNSVM+1

# 計算通過率
pass_rate = round(countNSVM/len(errvalue),2)
print('非線性SVM模型在門檻正負{}%內，通過率為{}%'.format(int(threshold*100),int(pass_rate*100)))

print('----------------------------------')

'''


############### XGBOOST ###############
# 建立 XGBOOST 模型
xgbr = XGBRegressor(
    learning_rate= 0.1,              # 每個迭代的學習率
    n_estimators = 39,               # 子模型數量，預設為100
    max_depth = 4,                   # 決策樹深度，預設為3
    min_child_weight = 3,
    gamma = 0,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = 'reg:squarederror',  # reg:squarederror、reg:linear(不推薦用)、binary:logistic
    reg_lambda = 0,
    reg_alpha = 0,
    nthread = 4,
    scale_pos_weight = 1,
    seed = 169
)

# 多輸出回歸，採用XGBOOST
mogr = MultiOutputRegressor(xgbr).fit(X_train, y_train)

# 訓練組
#xgbr.fit(X_train, y_train)
#y_pred_XG_train=xgbr.predict(X_train)

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

'''
# 把訓練好的模型儲存為pickle檔案 - DNN
with open('model/DNN.pickle','wb') as f:
   pickle.dump(nn,f)

# 把訓練好的模型儲存為pickle檔案 - 
with open('model/linear_svm.pickle','wb') as f:
   pickle.dump(linear_svm,f)

# 把訓練好的模型儲存為pickle檔案 - 
with open('model/svm.pickle','wb') as f:
   pickle.dump(svm,f)
'''

# 把訓練好的模型儲存為pickle檔案 - XGBOOST
with open('model/XGBOOST_muti.pickle','wb') as f:
   pickle.dump(mogr,f)

