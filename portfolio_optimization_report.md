# Portfolio Optimization Report
Generated on 2025-03-30 02:32:20

## Asset Information
Number of assets: 10
Top 5 assets by expected return:
- Asset_4: 0.12%
- Asset_1: 0.10%
- Asset_5: 0.09%
- Asset_3: 0.07%
- Asset_10: 0.03%

## Model Comparison
| Model | Type | Return | Risk | Sharpe | Assets |
|-------|------|--------|------|--------|--------|
| basic_mv | mean-variance | 0.11% | 0.74% | 0.15 | nan |
| constrained_mv | constrained-mean-variance | 0.11% | 0.74% | 0.15 | nan |
| integer_mv | integer-constrained | 0.11% | 0.73% | 0.15 | 3.0 |

## Model Details
### basic_mv
Model type: mean-variance
Status: solved
#### Performance Metrics
- expected_return: 0.11%
- portfolio_variance: 0.01%
- portfolio_volatility: 0.74%
- sharpe_ratio: 0.1501928362242954
- objective_value: 0.001004840933283919

#### Portfolio Composition
| Asset | Weight |
|-------|--------|
| Asset_4 | 56.32% |
| Asset_1 | 43.68% |

### constrained_mv
Model type: constrained-mean-variance
Status: solved
#### Performance Metrics
- expected_return: 0.11%
- portfolio_variance: 0.01%
- portfolio_volatility: 0.74%
- sharpe_ratio: 0.15019283521595042
- objective_value: 0.001004840933607693

#### Portfolio Composition
| Asset | Weight |
|-------|--------|
| Asset_4 | 56.32% |
| Asset_1 | 43.68% |

#### Active Constraints
- asset_bounds
- sector_constraints

### integer_mv
Model type: integer-constrained
Status: solved
#### Performance Metrics
- expected_return: 0.11%
- portfolio_variance: 0.01%
- portfolio_volatility: 0.73%
- sharpe_ratio: 0.15081370821822584
- objective_value: 0.0009972519562441444
- num_assets_selected: 3
- mip_gap: 0.007609889681582566

#### Portfolio Composition
| Asset | Weight |
|-------|--------|
| Asset_4 | 54.76% |
| Asset_1 | 40.24% |
| Asset_5 | 5.00% |

#### Active Constraints
- asset_bounds
- sector_constraints

#### Active Integer Constraints
- cardinality
- min_position_size
