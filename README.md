# RSRS

Research reproduction of Everbright Securities' Resistance-Support Relative Strength (RSRS) market-timing model. The indicator regresses rolling high prices on low prices and uses the slope, standardized slope, fit-adjusted score, and right-skewed score in historical trading rules.

## Implementations

`RSRS_code/` contains slope, Z-score, modified-score, and right-skewed-score scripts; transaction-cost and price/volume filters; and parameter-grid comparisons. Scripts specify the index, sample dates, regression window, standardization window, and entry/exit thresholds directly.

The long-only rules enter above the selected buy threshold and exit below the selected sell threshold. The code computes strategy returns and comparisons with the selected index. The index, dates, thresholds, regression/standardization windows, and cost settings are defined in each script.

## Reported project summary

Historical results reported for the configurations below:

| Index and period | Total return | Annualized return | Sharpe ratio | Maximum drawdown |
|---|---:|---:|---:|---:|
| CSI 300, 2005–2017 | 1,573.60% | 25.82% | 1.20 | 50.49% |
| SSE 50 | 1,432.36% | 24.84% | — | — |
| CSI 500 | 2,898.93% | 32.39% | — | — |

The project also reports a 75% correlation with future two-week returns. Forward-return correlation calculations are in `RSRS_code/RSRS_Modified_Zscore.py`, with the index, sample dates, and return window defined in the source.

## Configuration

Set `TUSHARE_TOKEN` in the Python process environment for Tushare-backed scripts. Local dataset paths and, in some scripts, an external `backtest` module directory are configured in the source. That helper module and the full market datasets are not included. The four-byte `RSRS+打分.py` file is an empty experiment entry, not an implemented scoring variant.

Reference: Everbright Securities' RSRS market-timing report. Contact: **tenghanz@usc.edu**.
