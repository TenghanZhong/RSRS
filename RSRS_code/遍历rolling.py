import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import tushare as ts
import os

# 初始化Tushare
pro = ts.pro_api('48c54212788b6a040d89de4ee5810744d936b44c2423302761f3b254')

# 数据处理函数

data_zscore = pro.index_daily(ts_code='000300.SH', start_date='20050101', end_date='20230301', fields=[
        "trade_date",
        "close",
        "high",
        "low"
    ])
data_zscore = data_zscore.sort_values(by='trade_date').fillna(method='ffill').dropna()
data_zscore['pct'] = data_zscore['close'] / data_zscore['close'].shift(1) - 1
data_zscore.dropna(inplace=True)

#保存文件到桌面
# 设置保存路径，将文件保存到桌面
desktop_path = os.path.expanduser(r"C:\Users\zth020906\Desktop")  # 获取桌面路径
csv_file_path = os.path.join(desktop_path, "data_zscore.csv")  # 合并路径和文件名

# 将data_zscore保存为CSV文件
data_zscore.to_csv(csv_file_path, index=False)

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
    if df.shape[0]<rolling:#行数小于18，返回NaN
        return np.nan
    y=df['high'].values#前18日窗口内的high
    x=df['low'].values.reshape(-1,1)
    lr=LinearRegression().fit(x,y)
    R_squared = lr.score(x, y)#前18日窗口的可决系数
    return R_squared

# 主循环


results = []  # 用于存储结果，每个元素是一个元组 (thre, N, total_ret)


for j in range(100,600,10):
    for s in range(14,19):
        data_zscore = pd.read_csv(r'C:\Users\zth020906\Desktop\data_zscore.csv')#直接用pd.read_csv不会对文件直接进行修改，每次循环都是原来文件
        data_zscore = data_zscore.set_index('trade_date')
        data_zscore['beta'] = [cal_beta(df, s) for df in data_zscore.rolling(s)]
        data_zscore['z_score'] = (data_zscore['beta'] - data_zscore['beta'].rolling(j).mean()) / data_zscore['beta'].rolling(j).std()
        data_zscore=data_zscore[data_zscore.index.astype('str')>'20150101']
        data_zscore = data_zscore.fillna(0)
        data_zscore['Modified_zscore'] = (data_zscore['z_score']) *([cal_R_squared(df,s) for df in data_zscore.rolling(s)])
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
        trade_times = 0  # 统计交易次数
        for i in range(0, data_zscore['right-skewed_z-score'].shape[0]):
            zscore = data_zscore.loc[i, 'right-skewed_z-score']
            if (position == 0) and zscore > buy_thre:
                # 无持仓买入
                data_zscore.loc[i, 'flag'] = 1
                data_zscore.loc[i + 1, 'position'] = 1
                position = 1
                trade_times += 1
            elif (position == 1) and (zscore < sell_thre):
                # 若之前有持仓，下穿卖出阈值则卖出
                data_zscore.loc[i, 'flag'] = -1
                data_zscore.loc[i + 1, 'position'] = 0  # 下一天position为0
                position = 0
                trade_times += 1
            else:  # 维持持仓
                data_zscore.loc[i + 1, 'position'] = data_zscore.loc[i, 'position']
        data_zscore.dropna(inplace=True)

        # 回测程序

        # RSRS策略的日收益率
        data_zscore['strategy_pct'] = data_zscore['pct'] * data_zscore['position']

        # 策略和沪深300的净值，累计收益率
        data_zscore['strategy'] = (1.0 + data_zscore['strategy_pct']).cumprod()  # cumprod是累乘
        data_zscore['hs300'] = (1.0 + data_zscore['pct']).cumprod()

        # 策略总收益率
        total_ret = 100 * (data_zscore['strategy'].iloc[-1] - 1)
        # 沪深300总收益率
        total_ret300 = 100 * (data_zscore['hs300'].iloc[-1] - 1)

        # 策略年化收益率
        annual_ret = 100 * (pow(1 + data_zscore['strategy'].iloc[-1],
                                250 / data_zscore['beta'].shape[0]) - 1)  # 得到年华收益数值*100  ，为了后面方便加上％

        # 沪深300年化收益率
        annual_ret300 = 100 * (pow(1 + data_zscore['hs300'].iloc[-1], 250 / data_zscore['hs300'].shape[0]) - 1)

        # 最大回撤率
        # MAX也可用累计收益率计算   因为pi=p0*ri pj=po*rj  所以max=max(ri-rj)/ri
        code = 'close'
        p_d = ((data_zscore['strategy'].cummax() - data_zscore['strategy']) / data_zscore['strategy'].cummax()).max()

        # 夏普比
        rets = (data_zscore['strategy'] / data_zscore['strategy'].shift(1) - 1).fillna(method='pad')
        exReturn = rets - 0.03 / 250
        sharperatio = np.sqrt(len(exReturn)) * exReturn.mean() / rets.std()

        # 存储结果
        results.append((s,j, total_ret))
        print('N={}，rolling={}已存储,收益率为{}%'.format(s,j,total_ret))
# 创建一个 Pandas DataFrame 来存储结果
results_df = pd.DataFrame(results, columns=['N', 'rolling', 'Total_Return'])

fig = plt.figure(figsize=(16, 8))
fig2 = plt.figure(figsize=(16, 8))
fig3 = plt.figure(figsize=(16, 8))

ax1 = fig.add_subplot(111)
rolling_max = results_df.groupby('rolling')['Total_Return'].max()
ax1.plot(rolling_max.index.values, rolling_max.values, marker='o', linestyle='-')
ax1.set_xlabel('rolling_max')
ax1.set_ylabel('Average Total Return (%)')
ax1.set_title('rolling_max vs. Average Total Return (Line Plot)')

