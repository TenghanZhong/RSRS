
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime
import qstock as qs


#数据处理
data=qs.get_data(code_list=['HS300'], start='20050125',end='20170301',freq='d')[['open','high','low','close']]
data = data.sort_index().fillna(method='ffill').dropna()
data['pct']=data['close']/data['close'].shift(1)-1
data.dropna(inplace=True)
data

# 回归求斜率--sklearn
def cal_beta(df,rolling=18):#咱只需要每个交易日滑动(rolling)计算前18个交易日最高价vs最低价的斜率
    if df.shape[0]<18:#行数小于18，返回NaN
        return np.nan
    y=df['high'].values#前18日窗口内的high
    x=df['low'].values.reshape(-1,1)
    lr=LinearRegression().fit(x,y)
    slope=lr.coef_[0]#前18日窗口内的斜率
    return slope
data['beta']=[cal_beta(df,18) for df in data.rolling(18)]
data.head(30)

#画beta分布图

plt.figure(figsize=(18, 8))
# 创建一个直方图
plt.hist(data['beta'], bins=100, color='blue', alpha=0.7)
# 添加标题和标签
plt.title('Distribution of Beta Values')
plt.xlabel('Beta Value')
plt.ylabel('Frequency')
custom_xticks = np.arange(0.39,1.38,0.05)
custom_xticks = np.round(custom_xticks, 2)
custom_xlabels = np.arange(0.39,1.38,0.05)
custom_xlabels = np.round(custom_xlabels, 2)
plt.xticks(custom_xticks, custom_xlabels)
# 显示柱状图
plt.show()


#计算beta统计性质
mean=data['beta'].mean()
std=data['beta'].std()
# 计算偏度（Skewness）
skew = data['beta'].skew()

# 计算峰度（Kurtosis）
kurt= data['beta'].kurtosis()

statistic=pd.DataFrame({'统计量' : ['mean', 'Standard Deviation', 'Skewness ', 'Kurtosis',],
                   '统计值' : [mean, std, skew, kurt]})
statistic=statistic.set_index('统计量')
print(statistic)

#交易策略编写
buy_thre=1.0
sell_thre=0.8
data1 = data.dropna().copy().reset_index(drop=False)#让最左列为1,2,3----索引,同时drop=True是删除掉索引列，False是会添加到columns
data1['flag'] = 0 # 买卖标记，买入：1，卖出：-1
data1['position'] = 0 # 持仓状态，持仓：1，不持仓：0
position=0#初始化
trade_times=0#统计交易次数
for i in range(0,data1['beta'].shape[0]):
    beta=data1.loc[i,'beta']
    if (position==0) and beta > buy_thre:
        #无持仓买入
        data1.loc[i,'flag']=1
        data1.loc[i+1,'position']=1
        position=1
        trade_times+=1
    elif (position == 1) and (beta < sell_thre):
        # 若之前有持仓，下穿卖出阈值则卖出
        data1.loc[i,'flag'] = -1
        data1.loc[i+1,'position'] = 0#下一天position为0
        position = 0
        trade_times+=1
    else:#维持持仓
        data1.loc[i+1,'position'] = data1.loc[i,'position']

#回测程序↓

# RSRS策略的日收益率
data1['strategy_pct'] = data1['pct'] * data1['position']

#策略和沪深300的净值，累计收益率
data1['strategy'] = (1.0 + data1['strategy_pct']).cumprod()#cumprod是累乘
data1['hs300'] = (1.0 + data1['pct']).cumprod()

#策略总收益率
total_ret=100*(data1['strategy'].iloc[-2]-1)
#沪深300总收益率
total_ret300=100*(data1['hs300'].iloc[-2]-1)

#策略年化收益率
annual_ret=100*(pow(1+data1['strategy'].iloc[-2],250/data1['beta'].shape[0])-1)#得到年华收益数值*100  ，为了后面方便加上％

#沪深300年化收益率
annual_ret300=100*(pow(1+data1['hs300'].iloc[-2],250/data1['beta'].shape[0])-1)

#最大回撤率
code='close'
p_d=((data1['strategy'].cummax()-data1['strategy'])/data1['strategy'].cummax()).max()

#夏普比
rets=(data1['strategy']/data1['strategy'].shift(1)-1).fillna(method='pad')
exReturn=rets-0.03/250
sharperatio=np.sqrt(len(exReturn))*exReturn.mean()/rets.std()

print('RSRS斜率量化择时策略的总收益率：%.2f%%，同期沪深300总收益率为:%.2f%%' % (total_ret, total_ret300))
print('RSRS斜率量化择时策略的年化收益率：%.2f%%，同期沪深300年化收益率为：%.2f%%' %(annual_ret,annual_ret300))
print('总交易次数:{}(买+卖)'.format(trade_times))
print(f'最大回撤率：{round(p_d*100,2)}%')
print(f'夏普比：{round(sharperatio,2)}')

data1.index = pd.to_datetime(data1['date'])#index换成日期

ax=data1[['strategy','hs300']].plot(figsize=(16,8),xlabel='时间',ylabel='净值',label=['策略收益','基准收益'],color=['SteelBlue','Red'],title='RSRS 指标策略在沪深 300 指数上的净值表现')
ax.legend(['策略收益','基准收益'])
plt.show()

#画沪深300斜率序列平均图
x1=data1['beta'].rolling(250).mean().plot(figsize=(16,8),xlabel='时间',ylabel='beta',color='r',label='沪深300斜率年平均')
data1['beta'].rolling(60).mean().plot(figsize=(16,8),xlabel='时间',ylabel='beta',color='SteelBlue',label='沪深300斜率季度平均')
x1.legend()
x1.set_title('沪深300斜率平均')
plt.show()


