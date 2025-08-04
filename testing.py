
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
    data = GetStockDataTest(tickers_swe['large_cap'],'2023-06-25','2025-06-25','1d',['Open','Close','High','Low','Volume'])
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

    #print(agent.evaluate_performance())
    #agent.plot_returns()
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

