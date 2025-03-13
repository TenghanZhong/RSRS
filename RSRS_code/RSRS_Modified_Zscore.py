###改进原理：
##我们使用线性拟合，并用斜率来量化支撑阻力相对强度。这样量化的方
#式使得所得斜率是否能够较有效地表达支撑阻力相对强度很大程度上受拟
#合本身效果的影响。在线性回归中，R 平方值（决定系数）可以理解成线性
#拟合效果的程度，取值在[0,1]区间，1 表示完全拟合。为了削弱拟合效果对
#策略的影响，我们通过将标准分值与决定系数相乘得到修正标准分。以更好削弱拟合差带来的效果影响



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime
import tushare as ts
pro = ts.pro_api('d689cb3c1d8c8a618e49ca0bb64f4d6de2f70e28ab5f76a867b31ac7')


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
data_zscore

#斜率标准分策略
def cal_beta(df,rolling=18):#咱只需要每个交易日滑动(rolling)计算前18个交易日最高价vs最低价的斜率
    if df.shape[0]<18:#行数小于18，返回NaN
        return np.nan
    y=df['high'].values#前18日窗口内的high
    x=df['low'].values.reshape(-1,1)
    lr=LinearRegression().fit(x,y)
    slope=lr.coef_[0]#前18日窗口内的斜率
    return slope

def cal_R_squared(df,rolling=18):#咱只需要每个交易日滑动(rolling)计算前18个交易日最高价vs最低价的斜率
    if df.shape[0]<18:#行数小于18，返回NaN
        return np.nan
    y=df['high'].values#前18日窗口内的high
    x=df['low'].values.reshape(-1,1)
    lr=LinearRegression().fit(x,y)
    R_squared = lr.score(x, y)#前18日窗口的可决系数
    return R_squared
data_zscore['beta']=[cal_beta(df,18) for df in data_zscore.rolling(18)]#迭代每个非空的18天窗口，计算18天的斜率贝塔
data_zscore['z_score']=(data_zscore['beta']-data_zscore['beta'].rolling(600,min_periods=20).mean())/data_zscore['beta'].rolling(600,min_periods=20).std()
data_zscore= data_zscore[17:]
data_zscore = data_zscore.fillna(0)
data_zscore['Modified_zscore']=(data_zscore['z_score'])*([cal_R_squared(df,18) for df in data_zscore.rolling(18)])
data_zscore.dropna(inplace=True)
data_zscore=data_zscore[2:]

#zscore分布图
plt.figure(figsize=(18, 8))
# 创建一个直方图
plt.hist(data_zscore['z_score'], bins=100, color='blue', alpha=0.7)
# 添加标题和标签
plt.title('Distribution of zscore Values')
plt.xlabel('zscore Value')
plt.ylabel('Frequency')
custom_xticks = np.arange(-4.9,3.7,0.4)
custom_xticks = np.round(custom_xticks, 2)
custom_xlabels = np.arange(-4.9,3.7,0.4)
custom_xlabels = np.round(custom_xlabels, 2)
plt.xticks(custom_xticks, custom_xlabels)
plt.show()


#画Modified_zscore分布图

plt.figure(figsize=(18, 8))
# 创建一个直方图画
plt.hist(data_zscore['Modified_zscore'], bins=100, color='blue', alpha=0.7)
# 添加标题和标签
plt.title('Distribution of Modified_zscore Values')
plt.xlabel('Modified_zscore Value')
plt.ylabel('Frequency')
custom_xticks = np.arange(-4.9,3.7,0.4)
custom_xticks = np.round(custom_xticks, 2)
custom_xlabels = np.arange(-4.9,3.7,0.4)
custom_xlabels = np.round(custom_xlabels, 2)
plt.xticks(custom_xticks, custom_xlabels)
plt.show()

#统计指标对比
mean_z,mean_mz=data_zscore['z_score'].mean(),data_zscore['Modified_zscore'].mean()
std_z,std_mz=data_zscore['z_score'].std(),data_zscore['Modified_zscore'].std()
skew_z,skew_mz = data_zscore['z_score'].skew(),data_zscore['Modified_zscore'].skew()
kurt_z,kurt_mz= data_zscore['z_score'].kurtosis(),data_zscore['Modified_zscore'].kurtosis()

statistic=pd.DataFrame({'统计量' : ['均值', '标准差', '偏度 ', '峰度'],
                   '标准分' : [mean_z, std_z, skew_z, kurt_z],
                        '修正标准分':[mean_mz, std_mz, skew_mz, kurt_mz]})
statistic=statistic.set_index('统计量')
print(statistic)

#交易策略编写
buy_thre=0.7
sell_thre=-0.7
data_zscore = data_zscore.reset_index(drop=False)#让最左列为1,2,3----索引,同时drop=True是删除掉索引列，False是会添加到columns
data_zscore['flag'] = 0 # 买卖标记，买入：1，卖出：-1
data_zscore['position'] = 0 # 持仓状态，持仓：1，不持仓：0
position=0#初始化
trade_times=0#统计交易次数
for i in range(0,data_zscore['Modified_zscore'].shape[0]):
    zscore=data_zscore.loc[i,'Modified_zscore']
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






#回测程序

# RSRS策略的日收益率
data_zscore['strategy_pct'] = data_zscore['pct'] * data_zscore['position']

#策略和沪深300的净值，累计收益率
data_zscore['strategy'] = (1.0 + data_zscore['strategy_pct']).cumprod()#cumprod是累乘
data_zscore['hs300'] = (1.0 + data_zscore['pct']).cumprod()

#策略总收益率
total_ret=100*(data_zscore['strategy'].iloc[-2]-1)
#沪深300总收益率
total_ret300=100*(data_zscore['hs300'].iloc[-2]-1)

#策略年化收益率
annual_ret=100*(pow(1+data_zscore['strategy'].iloc[-2],250/data_zscore['beta'].shape[0])-1)#得到年华收益数值*100  ，为了后面方便加上％

#沪深300年化收益率
annual_ret300=100*(pow(1+data_zscore['hs300'].iloc[-2],250/data_zscore['hs300'].shape[0])-1)

#最大回撤率
#MAX也可用累计收益率计算   因为pi=p0*ri pj=po*rj  所以max=max(ri-rj)/ri
code='close'
p_d=((data_zscore['strategy'].cummax()-data_zscore['strategy'])/data_zscore['strategy'].cummax()).max()

#夏普比
rets=(data_zscore['strategy']/data_zscore['strategy'].shift(1)-1).fillna(method='pad')
exReturn=rets-0.03/250
sharperatio=np.sqrt(len(exReturn))*exReturn.mean()/rets.std()

print('RSRS修正标准分策略的总收益率：%.2f%%，同期沪深300总收益率为:%.2f%%' % (total_ret, total_ret300))
print('RSRS修正标准分策略的年化收益率：%.2f%%，同期沪深300年化收益率为：%.2f%%' %(annual_ret,annual_ret300))
print('总交易次数:{}(买+卖)'.format(trade_times))
print(f'最大回撤率：{round(p_d*100,2)}%')
print(f'夏普比：{round(sharperatio,2)}')

data_zscore.index = pd.to_datetime(data_zscore['trade_date'])#index换成日期

ax=data_zscore[['strategy','hs300']].plot(figsize=(16,8),xlabel='时间',ylabel='净值',label=['策略收益','基准收益'],color=['SteelBlue','Red'],title='RSRS 标准分策略在沪深 300 指数上的净值表现')
ax.legend(['策略收益','基准收益'])
plt.show()



#判断标准分和未来十日预期收益关系
# 计算未来10天的收益率
data_zscore['future_return'] = data_zscore['close'].pct_change(periods=10).shift(-10)#把求得的预期十天收益率放在第一天，而非第10天

#pct_change是pd里的一个函数，用来计算增长率pct_change(periods='天数')

# 划分标准分区间
bins = np.arange(-3,4.2,0.3)
labels = np.arange(-2.85, 3.95, 0.3) # 区间标签,加0.15是为了将标签置于区间中央  如果你想将标签置于区间的中央，你可以在创建标签时加上区间宽度的一半。
data_zscore['z_score_bin'] = pd.cut(data_zscore['Modified_zscore'], bins=bins, labels=labels)

# 计算每个区间的平均收益率和上涨概率
mean_returns = data_zscore.groupby('z_score_bin')['future_return'].mean()
positive_probability = data_zscore.groupby('z_score_bin')['future_return'].apply(lambda x: (x > 0).mean())

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
left_corr_return = data_zscore[data_zscore['Modified_zscore'] < 0]['Modified_zscore'].corr(data_zscore[data_zscore['Modified_zscore'] < 0]['future_return'])
left_corr_probability =positive_probability[0:25]['z_score_bin'].corr(positive_probability[0:25]['future_return'])
#


# 计算右侧标准分值与市场未来期望收益的相关系数
right_corr_return = data_zscore[data_zscore['Modified_zscore'] >= 0]['Modified_zscore'].corr(data_zscore[data_zscore['Modified_zscore'] >= 0]['future_return'])
right_corr_probability = positive_probability[25:]['z_score_bin'].corr(positive_probability[25:]['future_return'])

# 计算整体标准分值与市场未来期望收益的相关系数
total_corr_return = data_zscore['Modified_zscore'].corr(data_zscore['future_return'])
total_corr_probability = positive_probability['z_score_bin'].corr(positive_probability['future_return'])

# 输出相关系数
print(f"左侧标准分值与市场未来期望收益的相关系数：{left_corr_return}")
print(f"左侧标准分值与上涨概率的相关系数：{left_corr_probability}")
print(f"右侧标准分值与市场未来期望收益的相关系数：{right_corr_return}")
print(f"右侧标准分值与上涨概率的相关系数：{right_corr_probability}")
print(f"整体标准分值与市场未来期望收益的相关系数：{total_corr_return}")
print(f"整体标准分值与上涨概率的相关系数：{total_corr_probability}")
