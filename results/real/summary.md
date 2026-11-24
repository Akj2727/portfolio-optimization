# Results

Out-of-sample period: 2013-04-11 onward (5 years, 59 monthly rebalances, 10 bp costs).

## Strategies

|                             | ann_return   | ann_vol   |   sharpe | max_drawdown   | cvar_95_daily   |
|:----------------------------|:-------------|:----------|---------:|:---------------|:----------------|
| Equal Weight                | 18.21%       | 14.59%    |     1.22 | -14.60%        | 2.20%           |
| GA-CVaR (stacked forecasts) | 12.77%       | 13.57%    |     0.95 | -13.91%        | 1.92%           |
| GA-MinCVaR (no forecasts)   | 12.76%       | 13.67%    |     0.95 | -13.66%        | 1.95%           |
| HMM-Blend                   | 18.96%       | 14.50%    |     1.27 | -13.05%        | 2.17%           |
| HMM-Switch                  | 28.23%       | 20.36%    |     1.32 | -16.31%        | 2.87%           |

## Forecast quality (mean cross-sectional rank IC, out of sample)

| Model | IC |
| --- | --- |
| rf | 0.1035 |
| xgb | 0.0948 |
| stack | -0.0450 |
| mom_126 (raw feature) | 0.1186 |

## GA tuning (equal evaluation budget, training data only)

Chosen: population 20, crossover 0.8, mutation 0.25.

|   pop_size |   crossover_rate |   mutation_rate |   gap_to_best |
|-----------:|-----------------:|----------------:|--------------:|
|  20.000000 |         0.800000 |        0.250000 |      0.000059 |
|  20.000000 |         0.950000 |        0.250000 |      0.000078 |
|  20.000000 |         0.600000 |        0.250000 |      0.000135 |
|  20.000000 |         0.800000 |        0.100000 |      0.000486 |
|  40.000000 |         0.950000 |        0.250000 |      0.000642 |
|  20.000000 |         0.950000 |        0.100000 |      0.000654 |
|  20.000000 |         0.600000 |        0.100000 |      0.000707 |
|  40.000000 |         0.800000 |        0.250000 |      0.000714 |
|  40.000000 |         0.600000 |        0.250000 |      0.000748 |
|  40.000000 |         0.950000 |        0.100000 |      0.001235 |
