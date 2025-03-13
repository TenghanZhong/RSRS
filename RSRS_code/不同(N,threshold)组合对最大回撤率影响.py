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

data_zscore = pro.index_daily(ts_code='000300.SH', start_date='20050101', end_date='20170301', fields=[
        "trade_date",
        "close",
        "high",
        "low"
    ])
data_zscore = data_zscore.sort_values(by='trade_date').fillna(method='ffill').dropna()
data_zscore = data_zscore.set_index('trade_date')
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


# 主循环
thres_values = np.arange(0.55, 0.95, 0.01)
N_values = range(15, 21)

results = []  # 用于存储结果，每个元素是一个元组 (thre, N, total_ret)

for N in N_values:
    for buy_thre in thres_values:
        data_zscore = pd.read_csv(r'C:\Users\zth020906\Desktop\data_zscore.csv')#直接用pd.read_csv不会对文件直接进行修改，每次循环都是原来文件
        data_zscore['beta'] = [cal_beta(df, N) for df in data_zscore.rolling(N)]
        data_zscore['z_score'] = (data_zscore['beta'] - data_zscore['beta'].rolling(600, min_periods=20).mean()) / data_zscore['beta'].rolling(600, min_periods=20).std()
        data_zscore.dropna(inplace=True)
        data_zscore = data_zscore.reset_index(drop=False)

        # 以下是你的交易策略和回测代码
        sell_thre = -buy_thre
        data_zscore['flag'] = 0  # 买卖标记，买入：1，卖出：-1
        data_zscore['position'] = 0  # 持仓状态，持仓：1，不持仓：0
        position = 0  # 初始化
        trade_times = 0  # 统计交易次数
        for i in range(0, data_zscore['z_score'].shape[0]):
            zscore = data_zscore.loc[i, 'z_score']
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
        Maximum_Drawdown=p_d

        # 夏普比
        rets = (data_zscore['strategy'] / data_zscore['strategy'].shift(1) - 1).fillna(method='pad')
        exReturn = rets - 0.03 / 250
        sharperatio = np.sqrt(len(exReturn)) * exReturn.mean() / rets.std()

        # 存储结果
        results.append((buy_thre, N, Maximum_Drawdown))
        print('组合({},{})已存储,最大回撤率为{}%'.format(N,buy_thre,Maximum_Drawdown*100))



# 创建一个 Pandas DataFrame 来存储结果
results_df = pd.DataFrame(results, columns=['Threshold', 'N', 'Maximum_Drawdown'])

# 使用 3D 散点图可视化结果
fig = plt.figure(figsize=(16, 8))
fig2 = plt.figure(figsize=(16, 8))
fig3 = plt.figure(figsize=(16, 8))
fig4 = plt.figure(figsize=(16, 8))


# 创建子图1：Threshold 对 Total Return 的影响
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(results_df['Threshold'], results_df['N'], results_df['Maximum_Drawdown'], c=results_df['Maximum_Drawdown'], cmap='viridis')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('N')
ax1.set_zlabel('Maximum_Drawdown (%)')
ax1.set_title('Threshold vs. Maximum_Drawdown')
ax1.auto_scale_xyz(results_df['Threshold'], results_df['N'], results_df['Maximum_Drawdown'])

# 创建子图2：N 对 Total Return 的影响
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(results_df['N'], results_df['Threshold'], results_df['Maximum_Drawdown'], c=results_df['Maximum_Drawdown'], cmap='viridis')
ax2.set_xlabel('N')
ax2.set_ylabel('Threshold')
ax2.set_zlabel('Maximum_Drawdown (%)')
ax2.set_title('N vs. Maximum_Drawdown')
ax2.auto_scale_xyz(results_df['Threshold'], results_df['N'], results_df['Maximum_Drawdown'])

# 创建子图3：N对 Total Return 的影响（线性图）
#把每个N对应的阈值均值回归了
ax3 = fig2.add_subplot(121)
threshold_means = results_df.groupby('N')['Maximum_Drawdown'].mean()
ax3.plot(threshold_means.index.values, threshold_means.values, marker='o', linestyle='-')
ax3.set_xlabel('N')
ax3.set_ylabel('Average Maximum_Drawdown (%)')
ax3.set_title('N vs. Average Maximum_Drawdown (Line Plot)')

# 创建子图4 threshold对 Total Return 的影响（线性图）

ax4 = fig2.add_subplot(122)
N_means = results_df.groupby('Threshold')['Maximum_Drawdown'].mean()
ax4.plot(N_means.index.values, N_means.values, marker='o', linestyle='-')
ax4.set_xlabel('Threshold')
ax4.set_ylabel('Average Maximum_Drawdown (%)')
ax4.set_title('Threshold vs. Average Maximum_Drawdown (Line Plot)')

# 创建子图5：N对 Total Return 的影响（线性图）
#把每个N对应的阈值均值回归了
ax5 = fig3.add_subplot(121)
threshold_means = results_df.groupby('N')['Maximum_Drawdown'].max()
ax5.plot(threshold_means.index.values, threshold_means.values, marker='o', linestyle='-')
ax5.set_xlabel('N')
ax5.set_ylabel('Average Maximum_Drawdown (%)')
ax5.set_title('N vs. MAX Maximum_Drawdown (Line Plot)')

# 创建子图6 threshold对 Total Return 的影响（线性图）

ax6 = fig3.add_subplot(122)
N_means = results_df.groupby('Threshold')['Maximum_Drawdown'].max()
ax6.plot(N_means.index.values, N_means.values, marker='o', linestyle='-')
ax6.set_xlabel('Threshold')
ax6.set_ylabel('Average Maximum_Drawdown (%)')
ax6.set_title('Threshold vs. MAX Maximum_Drawdown (Line Plot)')

# 创建子图5：Threshold 和 N 对 Total Return 的影响（曲面图）
ax6 = fig4.add_subplot(111, projection='3d')
threshold_values = results_df['Maximum_Drawdown'].values
n_values = results_df['N'].values
total_return_values = results_df['Maximum_Drawdown'].values
ax6.plot_trisurf(threshold_values, n_values, total_return_values, cmap='viridis', linewidth=0.2)
ax6.set_xlabel('Threshold')
ax6.set_ylabel('N')
ax6.set_zlabel('Maximum_Drawdown (%)')
ax6.set_title('Threshold and N vs. Maximum_Drawdown (Surface Plot)')
ax6.auto_scale_xyz(results_df['Threshold'], results_df['N'], results_df['Maximum_Drawdown'])
plt.show()
