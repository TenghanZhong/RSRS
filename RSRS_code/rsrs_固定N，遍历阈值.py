
###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime
import qstock as qs
import tushare as ts
pro = ts.pro_api('48c54212788b6a040d89de4ee5810744d936b44c2423302761f3b254')
#因为均值变动大，阈值变动大，改用标准分法
#当日标准分指标的计算方式：
#1. 取前 M 日的斜率时间序列。
#2. 以此样本计算当日斜率的标准分。 ---以前M日斜率时间序列样本均值和标准差当做u和sigma   然后标准分=(当日beta-u)/sigma
#3. 将计算得到的标准分 z 作为当日 RSRS 标准分指标值。

#数据处理
data_zscore=pro.index_daily(ts_code='000300.SH', start_date='20050101', end_date='20170301',fields=[
    "trade_date",
    "close",
    "high",
    "low"
])
data_zscore = data_zscore.sort_values(by='trade_date').fillna(method='ffill').dropna()
data_zscore=data_zscore.set_index('trade_date')
data_zscore['pct']=data_zscore['close']/data_zscore['close'].shift(1)-1
data_zscore.dropna(inplace=True)

#斜率标准分策略
def cal_beta(df,rolling=18):#咱只需要每个交易日滑动(rolling)计算前18个交易日最高价vs最低价的斜率
    if df.shape[0]<rolling:#行数小于18，返回NaN
        return np.nan
    y=df['high'].values#前18日窗口内的high
    x=df['low'].values.reshape(-1,1)
    lr=LinearRegression().fit(x,y)
    slope=lr.coef_[0]#前18日窗口内的斜率
    return slope
data_zscore['beta']=[cal_beta(df,18) for df in data_zscore.rolling(18)]#迭代每个非空的18天窗口，计算18天的斜率贝塔
data_zscore['z_score']=(data_zscore['beta']-data_zscore['beta'].rolling(600,min_periods=20).mean())/data_zscore['beta'].rolling(600,min_periods=20).std()
data_zscore.dropna(inplace=True)


buy_thres = []  # 存储不同的buy_thre值
total_returns = []  # 存储总收益率
trade_counts = []  # 存储总交易次数
max_drawdowns = []  # 存储最大回撤率
sharp_ratios = []  # 存储夏普比
data_zscore = data_zscore.reset_index(drop=False)#让最左列为1,2,3----索引,同时drop=True是删除掉索引列，False是会添加到columns

#交易策略编写
for buy_thre in np.arange(0.6, 0.81, 0.01).tolist():
    sell_thre=-buy_thre
    data_zscore['flag'] = 0 # 买卖标记，买入：1，卖出：-1
    data_zscore['position'] = 0 # 持仓状态，持仓：1，不持仓：0
    position=0#初始化
    trade_times=0#统计交易次数
    for i in range(0,data_zscore['z_score'].shape[0]):
        zscore=data_zscore.loc[i,'z_score']
        if (position==0) and zscore > buy_thre:
            #无持仓买入
            data_zscore.loc[i,'flag']=1
            data_zscore.loc[i+1,'position']=1
            position=1
            trade_times+=1
        elif (position == 1) and (zscore < sell_thre):
            # 若之前有持仓，下穿卖出阈值则卖出
            data_zscore.loc[i,'flag'] = -1
            data_zscore.loc[i+1,'position'] = 0#下一天position为0
            position = 0
            trade_times+=1
        else:#维持持仓
            data_zscore.loc[i+1,'position'] = data_zscore.loc[i,'position']
    data_zscore.dropna(inplace=True)


    #回测程序

    # RSRS策略的日收益率
    data_zscore['strategy_pct'] = data_zscore['pct'] * data_zscore['position']

    #策略和沪深300的净值，累计收益率
    data_zscore['strategy'] = (1.0 + data_zscore['strategy_pct']).cumprod()#cumprod是累乘
    data_zscore['hs300'] = (1.0 + data_zscore['pct']).cumprod()

    #策略总收益率
    total_ret=100*(data_zscore['strategy'].iloc[-1]-1)
    #沪深300总收益率
    total_ret300=100*(data_zscore['hs300'].iloc[-1]-1)

    #策略年化收益率
    annual_ret=100*(pow(1+data_zscore['strategy'].iloc[-1],250/data_zscore['beta'].shape[0])-1)#得到年华收益数值*100  ，为了后面方便加上％

    #沪深300年化收益率
    annual_ret300=100*(pow(1+data_zscore['hs300'].iloc[-1],250/data_zscore['hs300'].shape[0])-1)

    #最大回撤率
    #MAX也可用累计收益率计算   因为pi=p0*ri pj=po*rj  所以max=max(ri-rj)/ri
    code='close'
    p_d=((data_zscore['strategy'].cummax()-data_zscore['strategy'])/data_zscore['strategy'].cummax()).max()

    #夏普比
    rets=(data_zscore['strategy']/data_zscore['strategy'].shift(1)-1).fillna(method='pad')
    exReturn=rets-0.03/250
    sharperatio=np.sqrt(len(exReturn))*exReturn.mean()/rets.std()

    #添加数据
    buy_thres.append(buy_thre)
    total_returns.append(total_ret)
    trade_counts.append(trade_times)
    max_drawdowns.append(p_d)
    sharp_ratios.append(sharperatio)


# 绘制总收益率图
plt.figure(figsize=(12, 6))
plt.plot(buy_thres, total_returns, marker='o', linestyle='-')
plt.xlabel('Buy Threshold')
plt.ylabel('Total Returns (%)')
plt.title('Total Returns vs. Buy Threshold')
plt.grid(True)
plt.show()

# 绘制总交易次数图
plt.figure(figsize=(12, 6))
plt.plot(buy_thres, trade_counts, marker='o', linestyle='-')
plt.xlabel('Buy Threshold')
plt.ylabel('Total Trades')
plt.title('Total Trades vs. Buy Threshold')
plt.grid(True)
plt.show()

# 绘制最大回撤率图
plt.figure(figsize=(12, 6))
plt.plot(buy_thres, max_drawdowns, marker='o', linestyle='-')
plt.xlabel('Buy Threshold')
plt.ylabel('Max Drawdown (%)')
plt.title('Max Drawdown vs. Buy Threshold')
plt.grid(True)
plt.show()

# 绘制夏普比图
plt.figure(figsize=(12, 6))
plt.plot(buy_thres, sharp_ratios, marker='o', linestyle='-')
plt.xlabel('Buy Threshold')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio vs. Buy Threshold')
plt.grid(True)
plt.show()