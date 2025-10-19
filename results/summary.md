# Results

Out-of-sample period: 2020-08-31 onward (5 years, 59 monthly rebalances, 10 bp costs).

## Strategies

|                             | ann_return   | ann_vol   |   sharpe | max_drawdown   | cvar_95_daily   |
|:----------------------------|:-------------|:----------|---------:|:---------------|:----------------|
| Equal Weight                | 8.22%        | 20.80%    |     0.48 | -37.18%        | 3.12%           |
| GA-CVaR (stacked forecasts) | 5.61%        | 16.84%    |     0.41 | -35.53%        | 2.39%           |
| GA-MinCVaR (no forecasts)   | 5.68%        | 16.86%    |     0.41 | -35.37%        | 2.40%           |
| HMM-Blend                   | 9.22%        | 20.85%    |     0.53 | -35.59%        | 3.09%           |
| HMM-Switch                  | 9.86%        | 21.38%    |     0.55 | -35.01%        | 3.16%           |

## Forecast quality (mean cross-sectional rank IC, out of sample)

| Model | IC |
| --- | --- |
| rf | 0.0151 |
| xgb | -0.0267 |
| stack | 0.0423 |
| mom_126 (raw feature) | 0.0290 |

## GA tuning (equal evaluation budget, training data only)

Chosen: population 20, crossover 0.6, mutation 0.25.

|   pop_size |   crossover_rate |   mutation_rate |   gap_to_best |
|-----------:|-----------------:|----------------:|--------------:|
|  20.000000 |         0.600000 |        0.250000 |      0.000030 |
|  20.000000 |         0.800000 |        0.250000 |      0.000041 |
|  20.000000 |         0.950000 |        0.250000 |      0.000055 |
|  20.000000 |         0.600000 |        0.100000 |      0.000100 |
|  40.000000 |         0.600000 |        0.250000 |      0.000133 |
|  20.000000 |         0.800000 |        0.100000 |      0.000145 |
|  20.000000 |         0.950000 |        0.100000 |      0.000147 |
|  40.000000 |         0.800000 |        0.250000 |      0.000155 |
|  40.000000 |         0.950000 |        0.250000 |      0.000158 |
|  40.000000 |         0.600000 |        0.100000 |      0.000384 |
