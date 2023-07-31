import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


########## 原始及預測資料對比 ##########

table = pd.read_excel('output/future_pred.xlsx')
table['Date'] = pd.to_datetime(table['Date'])

# 區分訓練部分和預測部分
table_before = table[['Date','pred_revenue']]
table_before.loc[table_before['Date'] > '2023-07-15','pred_revenue'] = np.NaN

table_after = table[['Date','pred_revenue']]
table_after.loc[table_after['Date'] < '2023-07-15','pred_revenue'] = np.NaN

# 決定x軸及y軸
x = table['Date']
y1 = table['total_revenue']
y2 = table_before['pred_revenue']
y3 = table_after['pred_revenue']


# 建立畫布
fig = plt.figure()

# 使用額外的語言編碼避免中文亂碼
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']

# 繪製折線圖(也可以繪製多條)
plt.plot(x,y1,linewidth = 2,alpha = 1,label = '實際收益',linestyle='--',dashes=(1,1))
plt.plot(x,y2,linewidth = 1,alpha = 1,label = '模型訓練')
plt.plot(x,y3,linewidth = 1,alpha = 1,label = '預測收益',color = 'red')

plt.gcf().set_size_inches(16,8)                              # 設定圖表尺寸
plt.grid(True)                                               # 加入格線
plt.title('未來收益預測與實際比較',fontsize=15)                # 設定標題
plt.legend(loc='upper right',framealpha=.25,fontsize=12)     # 調整圖例設定

fig.savefig('output/未來收益預測與實際比較.png',transparent = False)  # 圖片存檔，決定背景是否透明
plt.show()