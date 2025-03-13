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
N_values = range(10, 20)

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

        # 夏普比
        rets = (data_zscore['strategy'] / data_zscore['strategy'].shift(1) - 1).fillna(method='pad')
        exReturn = rets - 0.03 / 250
        sharperatio = np.sqrt(len(exReturn)) * exReturn.mean() / rets.std()

        ##持仓总天数
        Total_Holding_Days = (data_zscore['position'] == 1).sum()

        ##平均持仓天数
        holding_periods = []  # 用于存储每次的持仓天数
        in_position = False  # 标记是否持仓
        entry_day = None  # 记录买入的日期
        for i in range(data_zscore.shape[0]):
            # 如果当前为买入信号
            if data_zscore.loc[i, 'flag'] == 1:
                entry_day = i
                in_position = True
            # 如果当前为卖出信号
            elif data_zscore.loc[i, 'flag'] == -1 and in_position:
                exit_day = i
                holding_period = exit_day - entry_day
                holding_periods.append(holding_period)
                in_position = False
        Average_Holding_Days = np.mean(holding_periods)

        ##持仓胜率（计算收益率为正的天数占所有持仓日的比例）
        holding_win_rate = data_zscore.groupby('position')['strategy_pct'].apply(lambda x: (x >= 0).mean())[
            1]  # 调取position=1时的胜率

        ##总胜率  总天数为空仓加持仓，然后胜的天数是空仓时期基准为负的天数加上持仓时期当日收益为正的天数
        ##要点：当两个Series使用&符号组合时，结果是一个新的布尔Series，其中只有在两个原始Series都为True的位置结果为True，其他地方为False。
        no_position_win_days = ((data_zscore['position'] == 0) & (data_zscore['pct'] < 0)).sum()  # 空仓时期基准 pct 为负的天数
        position_win_days = (
                    (data_zscore['position'] == 1) & (data_zscore['strategy_pct'] > 0)).sum()  # 持仓时期策略的日收益率为正的天数
        # 总的交易日数
        total_days = data_zscore.shape[0]
        # 总胜率
        Total_win_rate = (no_position_win_days + position_win_days) / total_days

        ##持仓期获利天数和持仓期亏损天数
        win_days = data_zscore.groupby('position')['strategy_pct'].apply(lambda x: (x >= 0).sum())[1]
        lose_days = data_zscore.groupby('position')['strategy_pct'].apply(lambda x: (x < 0).sum())[1]

        ## 盈利次数比，持仓期盈利次数÷持仓期总次数（buy_times=0）
        Profit_frequency = 0  # 用于存储盈利持仓次数
        Loss_frequency = 0  # 用于存储亏损持仓次数
        in_position = False  # 标记是否持仓
        start_close = 0  # 买入价格
        end_close = 0  # 卖出价格
        period_profit = 1  # 持仓期收益率
        Holding_Profit = []  # 用于存储持仓期收益率
        for i in range(data_zscore.shape[0]):
            # 如果当前为买入信号
            if data_zscore.loc[i, 'flag'] == 1:
                in_position = True
                start_close = data_zscore.loc[i, 'close']
            # 如果当前为卖出信号
            elif data_zscore.loc[i, 'flag'] == -1 and in_position:
                end_close = data_zscore.loc[i, 'close']
                period_profit = (end_close - start_close) / (start_close)
                Holding_Profit.append(period_profit)
                end_close, start_close = 0, 0  # 重置价格
                if period_profit >= 0:
                    Profit_frequency += 1
                    in_position = False
                    period_profit = 1  # 重置持仓期收益率
                else:
                    Loss_frequency += 1
                    in_position = False
                    period_profit = 1  # 重置持仓期收益率
        ## 如果循环结束后仍然持仓
        if in_position:
            end_close = data_zscore.loc[data_zscore.shape[0] - 2, 'close']
            period_profit = (end_close - start_close) / start_close
            Holding_Profit.append(period_profit)
            if period_profit >= 0:
                Profit_frequency += 1
            else:
                Loss_frequency += 1
        Profitability_ratio = Profit_frequency / (Profit_frequency + Loss_frequency)  # 盈利次数比

        # 存储结果
        # 存储结果
        results.append((buy_thre, N, total_ret, sharperatio, p_d, Total_win_rate,holding_win_rate, Profitability_ratio,win_days))  # 添加其他指标
        print('组合({},{})已存储,收益率为{}%,最大回撤率为{}%,总胜率为{}%,盈利期数比为{}%,'.format(N, buy_thre, total_ret, p_d,Total_win_rate*100,Profitability_ratio*100))




# 创建一个 Pandas DataFrame 来存储结果
results_df = pd.DataFrame(results, columns=['buy_thre', 'N', 'total_ret','sharperatio', 'p_d', 'Total_win_rate','holding_win_rate', 'Profitability_ratio','win_days'])  # 添加其他指标
# 绘制图表

# 总收益率
plt.figure(figsize=(10, 6))
for N in results_df['N'].unique():
    temp_df = results_df[results_df['N'] == N]
    plt.plot(temp_df['buy_thre'], temp_df['total_ret'], label=f'N={N}')
plt.xlabel('Buy Threshold')
plt.ylabel('Total Returns (%)')
plt.title('Total Returns at Different Buy Thresholds and N')
plt.legend()
plt.show()

# 夏普比率
plt.figure(figsize=(10, 6))
for N in results_df['N'].unique():
    temp_df = results_df[results_df['N'] == N]
    plt.plot(temp_df['buy_thre'], temp_df['sharperatio'], label=f'N={N}')
plt.xlabel('Buy Threshold')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio at Different Buy Thresholds and N')
plt.legend()
plt.show()

# 最大回撤
plt.figure(figsize=(10, 6))
for N in results_df['N'].unique():
    temp_df = results_df[results_df['N'] == N]
    plt.plot(temp_df['buy_thre'], temp_df['p_d'], label=f'N={N}')
plt.xlabel('Buy Threshold')
plt.ylabel('Max Drawdown (%)')
plt.title('Max Drawdown at Different Buy Thresholds and N')
plt.legend()
plt.show()

# 总胜率
plt.figure(figsize=(10, 6))
for N in results_df['N'].unique():
    temp_df = results_df[results_df['N'] == N]
    plt.plot(temp_df['buy_thre'], temp_df['Total_win_rate'], label=f'N={N}')
plt.xlabel('Buy Threshold')
plt.ylabel('Total Win Rate (%)')
plt.title('Total Win Rate at Different Buy Thresholds and N')
plt.legend()
plt.show()

# 其他指标可以使用类似的方式绘制

plt.tight_layout()
plt.show()





'''
# 使用 3D 散点图可视化结果
fig = plt.figure(figsize=(16, 8))
fig2 = plt.figure(figsize=(16, 8))
fig3 = plt.figure(figsize=(16, 8))
fig4 = plt.figure(figsize=(16, 8))


# 创建子图1：Threshold 对 Total Return 的影响
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(results_df['Threshold'], results_df['N'], results_df['Total_Return'], c=results_df['Total_Return'], cmap='viridis')
ax1.set_xlim(0.55,0.95)
ax1.set_ylim(15, 20)
ax1.set_zlim(400, 1400)
threshold_ticks = np.arange(0.6, 0.81, 0.02).tolist()
formatted_labels = ["{:.2f}".format(val) for val in threshold_ticks]
ax1.set_xticks(threshold_ticks)
ax1.set_xticklabels(formatted_labels)
n_ticks = [15, 16, 17, 18, 19, 20]
ax1.set_yticks(n_ticks)
ax1.set_yticklabels(['15', '16', '17', '18', '19', '20'])
ax1.set_xlabel('Threshold')
ax1.set_ylabel('N')
ax1.set_zlabel('Total Return (%)')
ax1.set_title('Threshold vs. Total Return')
ax1.auto_scale_xyz(results_df['Threshold'], results_df['N'], results_df['Total_Return'])

# 创建子图2：N 对 Total Return 的影响
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(results_df['N'], results_df['Threshold'], results_df['Total_Return'], c=results_df['Total_Return'], cmap='viridis')
ax2.set_xlim(15, 20)
ax2.set_ylim(0.55,0.95)
ax2.set_zlim(400, 1400)
n_ticks = [15, 16, 17, 18, 19, 20]
ax2.set_xticks(n_ticks)
ax2.set_xticklabels(['15', '16', '17', '18', '19', '20'])
threshold_ticks = np.arange(0.6, 0.81, 0.02).tolist()
formatted_labels = ["{:.2f}".format(val) for val in threshold_ticks]
ax2.set_yticks(threshold_ticks)
ax2.set_yticklabels(formatted_labels)
ax2.set_xlabel('N')
ax2.set_ylabel('Threshold')
ax2.set_zlabel('Total Return (%)')
ax2.set_title('N vs. Total Return')
ax2.auto_scale_xyz(results_df['Threshold'], results_df['N'], results_df['Total_Return'])

# 创建子图3：N对 Total Return 的影响（线性图）
#把每个N对应的阈值均值回归了
ax3 = fig2.add_subplot(121)
threshold_means = results_df.groupby('N')['Total_Return'].mean()
ax3.plot(threshold_means.index.values, threshold_means.values, marker='o', linestyle='-')
ax3.set_xlabel('N')
ax3.set_ylabel('Average Total Return (%)')
ax3.set_title('N vs. Average Total Return (Line Plot)')

# 创建子图4 threshold对 Total Return 的影响（线性图）

ax4 = fig2.add_subplot(122)
N_means = results_df.groupby('Threshold')['Total_Return'].mean()
ax4.plot(N_means.index.values, N_means.values, marker='o', linestyle='-')
ax4.set_xlabel('Threshold')
ax4.set_ylabel('Average Total Return (%)')
ax4.set_title('Threshold vs. Average Total Return (Line Plot)')

# 创建子图5：N对 Total Return 的影响（线性图）
#把每个N对应的阈值均值回归了
ax5 = fig3.add_subplot(121)
threshold_means = results_df.groupby('N')['Total_Return'].max()
ax5.plot(threshold_means.index.values, threshold_means.values, marker='o', linestyle='-')
ax5.set_xlabel('N')
ax5.set_ylabel('Average Total Return (%)')
ax5.set_title('N vs. MAX Total Return (Line Plot)')

# 创建子图6 threshold对 Total Return 的影响（线性图）

ax6 = fig3.add_subplot(122)
N_means = results_df.groupby('Threshold')['Total_Return'].max()
ax6.plot(N_means.index.values, N_means.values, marker='o', linestyle='-')
ax6.set_xlabel('Threshold')
ax6.set_ylabel('Average Total Return (%)')
ax6.set_title('Threshold vs. MAX Total Return (Line Plot)')

# 创建子图5：Threshold 和 N 对 Total Return 的影响（曲面图）
ax6 = fig4.add_subplot(111, projection='3d')
threshold_values = results_df['Threshold'].values
n_values = results_df['N'].values
total_return_values = results_df['Total_Return'].values
ax6.plot_trisurf(threshold_values, n_values, total_return_values, cmap='viridis', linewidth=0.2)
ax6.set_xlim(0.55, 0.95)
ax6.set_ylim(15, 20)
ax6.set_xlabel('Threshold')
ax6.set_ylabel('N')
ax6.set_zlabel('Total Return (%)')
ax6.set_title('Threshold and N vs. Total Return (Surface Plot)')
ax6.auto_scale_xyz(results_df['Threshold'], results_df['N'], results_df['Total_Return'])
plt.show()
'''
