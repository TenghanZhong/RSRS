import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import tushare as ts

# 设置字体，例如使用DejaVu Sans字体（支持大多数Unicode字符）
plt.rcParams['font.sans-serif'] = ['SimHei']

# 也可以设置字体大小
plt.rcParams['font.size'] = 12
# 初始化Tushare
pro = ts.pro_api('48c54212788b6a040d89de4ee5810744d936b44c2423302761f3b254')

# 数据处理函数
ts_code='000300.SH'
data_zscore = pro.index_daily(ts_code=ts_code, start_date='20050101', end_date='20170431', fields=[
        "trade_date",
        "close",
        "high",
        "low"
    ])
data_zscore = data_zscore.sort_values(by='trade_date').fillna(method='ffill').dropna()
data_zscore = data_zscore.set_index('trade_date')
data_zscore['pct'] = data_zscore['close'] / data_zscore['close'].shift(1) - 1
data_zscore.dropna(inplace=True)



# 计算beta值函数
def cal_beta(df, rolling):
    if df.shape[0] < rolling:
        return np.nan
    y = df['high'].values
    x = df['low'].values.reshape(-1, 1)
    lr = LinearRegression().fit(x, y)
    slope = lr.coef_[0]
    return slope
def cal_R_squared(df,rolling):#咱只需要每个交易日滑动(rolling)计算前18个交易日最高价vs最低价的斜率
    if df.shape[0]<rolling:
        return np.nan
    y=df['high'].values#前18日窗口内的high
    x=df['low'].values.reshape(-1,1)
    lr=LinearRegression().fit(x,y)
    R_squared = lr.score(x, y)#前18日窗口的可决系数
    return R_squared

data_zscore['MA20'] = data_zscore['close'].rolling(20).mean()
data_zscore['beta'] = [cal_beta(df, 16) for df in data_zscore.rolling(16)]
data_zscore['z_score'] = (data_zscore['beta'] - data_zscore['beta'].rolling(300,min_periods=20).mean()) / data_zscore['beta'].rolling(300,min_periods=20).std()
data_zscore = data_zscore[15:]#第15列开始beta不为0
data_zscore = data_zscore.fillna(0)
data_zscore['Modified_zscore'] = (data_zscore['z_score']) *([cal_R_squared(df,16) for df in data_zscore.rolling(16)])
data_zscore.dropna(inplace=True)
data_zscore = data_zscore[2:]
data_zscore['right-skewed_z-score'] = data_zscore['beta'] * data_zscore['Modified_zscore']
data_zscore = data_zscore.reset_index(drop=False)

# 以下是你的交易策略和回测代码
buy_thre = 0.7
sell_thre=-0.7
data_zscore['flag'] = 0  # 买卖标记，买入：1，卖出：-1
data_zscore['position'] = 0  # 持仓状态，持仓：1，不持仓：0
position = 0  # 初始化
data_zscore['trade_times'] = 0  # 统计交易次数
buy_times=0
trade_times=0# 统计交易次数
for i in range(0, data_zscore['right-skewed_z-score'].shape[0]):
    zscore = data_zscore.loc[i, 'right-skewed_z-score']
    if ((position == 0) and zscore > buy_thre and data_zscore.loc[i-1, 'MA20']>data_zscore.loc[i-3, 'MA20']):#过滤左侧交易
        # 无持仓买入
        data_zscore.loc[i, 'flag'] = 1
        data_zscore.loc[i + 1, 'position'] = 1
        position = 1
        trade_times += 1
        data_zscore.loc[i, 'trade_times'] = 1
    elif (position == 1) and (zscore < sell_thre):
        # 若之前有持仓，下穿卖出阈值则卖出
        data_zscore.loc[i, 'flag'] = -1
        data_zscore.loc[i + 1, 'position'] = 0  # 下一天position为0
        position = 0
        trade_times += 1
        data_zscore.loc[i, 'trade_times'] = 1
    else:  # 维持持仓
        data_zscore.loc[i + 1, 'position'] = data_zscore.loc[i, 'position']
data_zscore.dropna(inplace=True)

# 回测程序
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
data_zscore['z_score_bin'] = pd.cut(data_zscore['right-skewed_z-score'], bins=bins, labels=labels)

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
ax1.set_xlabel('右偏标准分区间')
ax1.set_ylabel('平均收益率')
ax1.set_xticks(labels)
ax1.set_title('右偏标准分值与未来市场预期收益率的关系')

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

