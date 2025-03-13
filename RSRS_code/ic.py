import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from alphalens import utils
from alphalens import plotting
from alphalens import tears
from alphalens import performance

def test_ic(data_zscore,ts_code):
    data_zscore['trade_date'] = pd.to_datetime(data_zscore['trade_date'])  # index换成日期
    ts_code_list = []
    ts_code_list.append(ts_code)
    # 创建随机因子数据
    factor_data = pd.DataFrame({
        'date': data_zscore['trade_date'],
        'asset': np.tile(ts_code, len(data_zscore['trade_date'])),
        'factor': data_zscore['z_score']
    })
    factor_data.dropna(inplace=True)
    # 设置索引
    factor_data.set_index(['date', 'asset'], inplace=True)
    # 设置收盘价序列
    prices = pd.DataFrame({
        'date': data_zscore['trade_date'],
        ts_code: data_zscore['close']
    })
    prices.set_index('date', inplace=True)
    prices.dropna(inplace=True)
    factor_data_cleaned = utils.get_clean_factor_and_forward_returns(factor=factor_data, prices=prices,
                                                                     quantiles=None, bins=5)
    tears.create_full_tear_sheet(factor_data_cleaned)