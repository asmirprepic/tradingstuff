
    # %%
    import numpy as np

    # %%
    %load_ext autoreload
    %autoreload 2

    # %%
    from data_handling.get_stock_data import GetStockDataTest
    from data_handling.get_stock_tickers import GetTickers

    # %%
    getTickers = GetTickers()


    # %%
    tickers_swe = getTickers._processTickers(region = 'se')
    # %%
    data = GetStockDataTest(tickers_swe['large_cap'],'2023-06-25','2025-08-10','1d',['Open','Close','High','Low','Volume'])
    # %%
    large_cap_se = data.getData()

    # %%
    from agents.technical.volume_price_divergence_agent import VolumePriceDivergenceAgent
    vpd_agent = VolumePriceDivergenceAgent(large_cap_se)

    # %%
    vpd_agent.plot_returns()

    # %%
    signals = vpd_agent.signal_data['AAK.ST']

    # %%
    print("Signal distribution:\n", signals['Signal'].value_counts())
    print("Avg return after signal:\n", signals.groupby('Signal')['return'].mean())

    # %%
    from agents.ml_based.clustering_agent import ClusteringFilteredKNNAgent

    # %%

    agent = ClusteringFilteredKNNAgent(large_cap_se, n_neighbors=10, n_clusters=3, cluster_lookback=20)
    agent.filter_stocks_by_cluster()

    print(agent.evaluate_performance())
    agent.plot_returns()
    # %%
    agent.run_all(mode='backtest')
    # %%
    from agents.ml_based.knn_agent import KNNAgent

    # %%
    test_KNN = KNNAgent(large_cap_se,15)

    # %%
    test_KNN.run_all_walk_forward()

    # %%
    eval_knn = test_KNN.evaluate_performance()

    # %%
    from agents.ml_based.logistic_reg_agent import LRAgent

    # %%
    test_LR = LRAgent(large_cap_se)

    # %%
    test_LR.run_all_walk_forward()

    # %%
    eval_LR = test_LR.evaluate_performance()


    # %%
    from agents.technical.momentum_agent import MomentumAgent

    mom_agent = MomentumAgent(large_cap_se,back_length=5)

    # %%
    agent = ClusteringFilteredKNNAgent(
    data=large_cap_se,
    n_neighbors=10,
    n_clusters=3,
    cluster_lookback=20
    )

    # %%

    # 2️⃣ Pick the stocks via clustering
    agent.filter_stocks_by_cluster(as_of_date="2025-06-01")
    # if you skip as_of_date, it will cluster using the entire dataset
    # %%

    # 3️⃣ Run only on filtered stocks
    agent.run_filtered(mode="backtest")

# %%
stock = agent.filtered_stocks[0]

X, Y = agent.feature_engineering(stock)

print("Any NaNs in X? ->", X.isna().any().any())
print("NaNs per feature:\n", X.isna().sum())
print("First 10 rows with any NaN:\n", X[X.isna().any(axis=1)].head(10))
