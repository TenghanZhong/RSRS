#改进原理：很多发表的研究表明市场涨跌与交易量有明显的正相关性。借鉴类似的想法，
# 我们尝试用交易量与修正标准分之间的相关性来过滤误判信号。只有在相关性为正的时刻给出的交易信号，我们才认为是
# 合理的信号。
###这一方法可以过滤误判信号，减少回撤，增大超额收益

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
ts_code='000300.SH'
#数据处理   #000300.SH沪深300指数  000905.SH中证指数 000941.SH新能源指数
data_zscore=pro.index_daily(ts_code=ts_code, start_date='20050101', end_date='20170401',fields=[
    "trade_date",
    "close",
    "high",
    "low",
    'vol'
])
data_zscore = data_zscore.sort_values(by='trade_date').fillna(method='ffill').dropna()
data_zscore=data_zscore.set_index('trade_date')
data_zscore['pct']=data_zscore['close']/data_zscore['close'].shift(1)-1
data_zscore.dropna(inplace=True)

#斜率标准分策略
def cal_beta(df,rolling=18):#咱只需要每个交易日滑动(rolling)计算前18个交易日最高价vs最低价的斜率
    if df.shape[0]<18:#行数小于18，返回NaN
        return np.nan
    y=df['high'].values#前18日窗口内的high
    x=df['low'].values.reshape(-1,1)
    lr=LinearRegression().fit(x,y)
    slope=lr.coef_[0]#前18日窗口内的斜率
    return slope
data_zscore['beta']=[cal_beta(df,18) for df in data_zscore.rolling(18)]#迭代每个非空的18天窗口，计算18天的斜率贝塔
data_zscore['z_score']=(data_zscore['beta']-data_zscore['beta'].rolling(600,min_periods=20).mean())/data_zscore['beta'].rolling(600,min_periods=20).std()
data_zscore.dropna(inplace=True)
data_zscore = data_zscore.reset_index(drop=False)

#交易策略编写
buy_thre = 0.7
sell_thre=-0.7
data_zscore['flag'] = 0  # 买卖标记，买入：1，卖出：-1
data_zscore['position'] = 0  # 持仓状态，持仓：1，不持仓：0
position = 0  # 初始化
data_zscore['trade_times'] = 0  # 统计交易次数
buy_times=0
trade_times=0# 统计交易次数
for i in range(0,data_zscore['z_score'].shape[0]):
    if i<11:#行数小于18，返回NaN
        corr = 0
    else:
        zscore=data_zscore.loc[i,'z_score']
        corr= data_zscore[i-11:i-1]['vol'].corr(data_zscore[i-11:i-1]['z_score'])#前 10 日交易量与标准分之间的相关性
        if (position==0) and zscore > buy_thre and corr>0:
            #无持仓买入
            data_zscore.loc[i,'flag']=1
            data_zscore.loc[i+1,'position']=1
            position=1
            trade_times+=1
            data_zscore.loc[i, 'trade_times'] = 1
        elif (position == 1) and (zscore < sell_thre):
            # 若之前有持仓，下穿卖出阈值则卖出
            data_zscore.loc[i,'flag'] = -1
            data_zscore.loc[i+1,'position'] = 0#下一天position为0
            position = 0
            trade_times+=1
            data_zscore.loc[i, 'trade_times'] = 1
        else:#维持持仓
            data_zscore.loc[i+1,'position'] = data_zscore.loc[i,'position']
data_zscore.dropna(inplace=True)

#回测程序
from 回测框架 import run_backtest
# 运行回测
results = run_backtest(data_zscore,ts_code)
# 打印或处理结果
print(results)



#判断标准分和未来十日预期收益关系
# 计算未来10天的收益率
data_zscore['future_return'] = data_zscore['close'].pct_change(periods=10).shift(-10)#把求得的预期十天收益率放在第一天，而非第10天

#pct_change是pd里的一个函数，用来计算增长率pct_change(periods='天数')

# 划分标准分区间
bins = np.arange(-5,4.2,0.2)
labels = np.arange(-5,4,0.2)+ 0.1  # 区间标签,加0.1是为了将标签置于区间中央
data_zscore['z_score_bin'] = pd.cut(data_zscore['z_score'], bins=bins, labels=labels)

# 计算每个区间的平均收益率和上涨概率
mean_returns = data_zscore.groupby('z_score_bin')['future_return'].mean()
positive_probability = data_zscore.groupby('z_score_bin')['future_return'].apply(lambda x: (x > 0).mean())
#lambda x: (x > 0).mean(): 这是一个lambda函数，用于计算每个分组中'future_return'列中大于0的值所占的比例。
# 在lambda函数内部，(x > 0)是一个布尔数组，表示'future_return'列中哪些值大于0，
# 然后.mean()函数计算了这个布尔数组中True值（即大于0的值）的比例，即大于0的值所占的比例。


# 绘制图表
fig, ax1 = plt.subplots(figsize=(16, 8))

# 绘制平均收益率柱状图
ax1.bar(mean_returns.index, mean_returns.values, color='b', alpha=0.7, label='平均收益率')
ax1.set_xlabel('标准分区间')
ax1.set_ylabel('平均收益率')
ax1.set_xticks(labels)
ax1.set_title('标准分值与未来市场预期收益率的关系')

# 创建第二个y轴，绘制上涨概率线图
ax2 = ax1.twinx()
ax2.plot(positive_probability.index, positive_probability.values, color='r', marker='o', label='上涨概率')
ax2.set_ylabel('上涨概率')

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()

positive_probability=pd.DataFrame(positive_probability)#转换格式为dataframe
positive_probability=positive_probability.reset_index(drop=False)

##处理NaN值的方法通常是使用.corr()函数，它会自动忽略包含NaN值的行

# 计算左侧标准分值与市场未来期望收益的相关系数
left_corr_return = data_zscore[data_zscore['z_score'] < 0]['z_score'].corr(data_zscore[data_zscore['z_score'] < 0]['future_return'])
left_corr_probability =positive_probability[0:25]['z_score_bin'].corr(positive_probability[0:25]['future_return'])
#


# 计算右侧标准分值与市场未来期望收益的相关系数
right_corr_return = data_zscore[data_zscore['z_score'] >= 0]['z_score'].corr(data_zscore[data_zscore['z_score'] >= 0]['future_return'])
right_corr_probability = positive_probability[25:]['z_score_bin'].corr(positive_probability[25:]['future_return'])

# 计算整体标准分值与市场未来期望收益的相关系数
total_corr_return = data_zscore['z_score'].corr(data_zscore['future_return'])
total_corr_probability = positive_probability['z_score_bin'].corr(positive_probability['future_return'])

# 输出相关系数
print(f"左侧标准分值与市场未来期望收益的相关系数：{left_corr_return}")
print(f"左侧标准分值与上涨概率的相关系数：{left_corr_probability}")
print(f"右侧标准分值与市场未来期望收益的相关系数：{right_corr_return}")
print(f"右侧标准分值与上涨概率的相关系数：{right_corr_probability}")
print(f"整体标准分值与市场未来期望收益的相关系数：{total_corr_return}")
print(f"整体标准分值与上涨概率的相关系数：{total_corr_probability}")