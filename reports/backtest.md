# Backtest: 12 rolling-origin folds, step 14 days, horizons [7, 14, 28]

Data: daily arrivals per hotel, 2015-07-01 to 2017-08-31; the fold origins run from 2017-03-03 to 2017-08-04.
Champion by pooled WAPE at h=14 among models beating seasonal naive at every horizon: **sarima**.

## Pooled over both hotels (WAPE = total absolute error / total arrivals; lower is better)

| Model | h | WAPE | vs seasonal naive | MAPE | MASE | beats seasonal naive |
|---|---|---|---|---|---|---|
| sarima | 7 | 21.4% | 0.74x | 26.0% | 0.65 | yes |
| ets | 7 | 22.5% | 0.77x | 27.4% | 0.68 | yes |
| lightgbm | 7 | 23.6% | 0.81x | 30.3% | 0.73 | yes |
| prophet | 7 | 24.5% | 0.84x | 30.8% | 0.75 | yes |
| seasonal_naive | 7 | 29.1% | 1.00x | 33.7% | 0.88 | no |
| naive | 7 | 37.5% | 1.29x | 51.6% | 1.18 | no |
| sarima | 14 | 21.5% | 0.74x | 25.6% | 0.65 | yes |
| ets | 14 | 22.4% | 0.77x | 27.2% | 0.68 | yes |
| lightgbm | 14 | 23.5% | 0.81x | 29.3% | 0.72 | yes |
| prophet | 14 | 23.7% | 0.82x | 30.0% | 0.72 | yes |
| seasonal_naive | 14 | 29.0% | 1.00x | 32.9% | 0.88 | no |
| naive | 14 | 36.5% | 1.26x | 48.9% | 1.16 | no |
| sarima | 28 | 21.8% | 0.71x | 25.6% | 0.66 | yes |
| ets | 28 | 22.9% | 0.75x | 27.6% | 0.69 | yes |
| lightgbm | 28 | 23.2% | 0.75x | 28.3% | 0.70 | yes |
| prophet | 28 | 23.9% | 0.78x | 30.0% | 0.73 | yes |
| seasonal_naive | 28 | 30.7% | 1.00x | 34.5% | 0.93 | no |
| naive | 28 | 36.9% | 1.20x | 49.5% | 1.17 | no |

## City Hotel

| Model | h | WAPE (pooled) | WAPE (mean of folds) | vs seasonal naive | MAPE | MASE | fit s |
|---|---|---|---|---|---|---|---|
| sarima | 7 | 18.0% | 17.8% | 0.70x | 18.4% | 0.58 | 0.59 |
| lightgbm | 7 | 18.9% | 18.8% | 0.74x | 19.9% | 0.61 | 1.21 |
| ets | 7 | 19.0% | 18.9% | 0.74x | 19.8% | 0.61 | 0.37 |
| prophet | 7 | 20.4% | 20.3% | 0.79x | 21.3% | 0.66 | 0.32 |
| seasonal_naive | 7 | 25.7% | 25.7% | 1.00x | 27.5% | 0.83 | 0.00 |
| naive | 7 | 27.1% | 27.1% | 1.06x | 30.4% | 0.88 | 0.00 |
| sarima | 14 | 18.6% | 18.4% | 0.73x | 19.3% | 0.61 | 0.59 |
| ets | 14 | 19.2% | 19.0% | 0.75x | 20.5% | 0.63 | 0.37 |
| prophet | 14 | 19.9% | 19.8% | 0.78x | 21.0% | 0.65 | 0.32 |
| lightgbm | 14 | 19.9% | 19.7% | 0.78x | 21.0% | 0.65 | 1.21 |
| seasonal_naive | 14 | 25.4% | 25.3% | 1.00x | 27.1% | 0.83 | 0.00 |
| naive | 14 | 26.0% | 25.9% | 1.02x | 29.7% | 0.85 | 0.00 |
| sarima | 28 | 19.0% | 18.9% | 0.70x | 19.6% | 0.62 | 0.59 |
| ets | 28 | 20.0% | 19.9% | 0.74x | 21.4% | 0.65 | 0.37 |
| lightgbm | 28 | 20.1% | 20.0% | 0.74x | 21.3% | 0.66 | 1.21 |
| prophet | 28 | 20.4% | 20.4% | 0.75x | 21.6% | 0.67 | 0.32 |
| naive | 28 | 26.5% | 26.5% | 0.98x | 30.2% | 0.86 | 0.00 |
| seasonal_naive | 28 | 27.1% | 27.1% | 1.00x | 28.5% | 0.88 | 0.00 |

## Resort Hotel

| Model | h | WAPE (pooled) | WAPE (mean of folds) | vs seasonal naive | MAPE | MASE | fit s |
|---|---|---|---|---|---|---|---|
| sarima | 7 | 28.0% | 28.6% | 0.78x | 33.7% | 0.73 | 0.41 |
| ets | 7 | 29.2% | 29.7% | 0.82x | 35.0% | 0.76 | 0.19 |
| prophet | 7 | 32.3% | 33.3% | 0.90x | 40.3% | 0.84 | 0.30 |
| lightgbm | 7 | 32.5% | 33.4% | 0.91x | 40.8% | 0.84 | 1.31 |
| seasonal_naive | 7 | 35.7% | 36.1% | 1.00x | 39.8% | 0.92 | 0.00 |
| naive | 7 | 57.1% | 58.4% | 1.60x | 72.8% | 1.48 | 0.00 |
| sarima | 14 | 27.1% | 27.0% | 0.75x | 31.9% | 0.70 | 0.41 |
| ets | 14 | 28.5% | 28.4% | 0.79x | 33.8% | 0.74 | 0.19 |
| lightgbm | 14 | 30.4% | 30.3% | 0.85x | 37.6% | 0.79 | 1.31 |
| prophet | 14 | 31.0% | 31.0% | 0.86x | 39.0% | 0.80 | 0.30 |
| seasonal_naive | 14 | 35.9% | 35.6% | 1.00x | 38.6% | 0.93 | 0.00 |
| naive | 14 | 56.7% | 56.8% | 1.58x | 68.2% | 1.47 | 0.00 |
| sarima | 28 | 27.0% | 26.9% | 0.72x | 31.6% | 0.70 | 0.41 |
| ets | 28 | 28.5% | 28.4% | 0.76x | 33.8% | 0.73 | 0.19 |
| lightgbm | 28 | 29.1% | 28.9% | 0.77x | 35.3% | 0.75 | 1.31 |
| prophet | 28 | 30.5% | 30.3% | 0.81x | 38.4% | 0.79 | 0.30 |
| seasonal_naive | 28 | 37.5% | 37.3% | 1.00x | 40.5% | 0.97 | 0.00 |
| naive | 28 | 56.9% | 58.1% | 1.52x | 68.9% | 1.47 | 0.00 |

Method: every model is refit on all data before each fold's origin and forecasts the next 28 days; the first 7, 14 and 28
days are scored. MAPE skips days with zero arrivals; WAPE pools absolute errors over the window. MASE scales by the
in-sample seasonal-naive (weekly) error. Nothing is tuned on the test windows.
