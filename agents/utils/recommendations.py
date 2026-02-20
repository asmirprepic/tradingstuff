import pandas as pd


def recommendations_from_agent(agent, persistence=1, min_score=None, top_n=None, save_path=None):
    """Generate ranked Buy/Sell/Hold recommendations from a trading agent.

    Args:
        agent: An instance of `TradingAgent` (or subclass) with `signal_data` and `returns_data`.
        persistence (int): Require the same `Position` (target exposure) for the last `persistence` rows
                           to accept the recommendation. If not satisfied the recommendation is set to
                           'Hold'. Default 1.
        min_score (float or None): If set, filter out stocks with `Score` below this value.
        top_n (int or None): If set, keep only the top N stocks by `Score`.
        save_path (str or None): If provided, save the resulting DataFrame to this CSV path.

    Returns:
        pd.DataFrame: DataFrame with columns:
          - Stock
          - Recommendation (Buy/Sell/Hold)
          - Position (target exposure: -1/0/1)
          - TradeNow (bool; True if last Signal != 0)
          - Signal (last trade event: -1/0/1 if present else 0)
          - Score (strategy score, typically % log-return)
        sorted by Score desc.
    """

    rows = []

    for stock, signals in agent.signal_data.items():
        if signals is None or signals.empty:
            continue

        # Ensure Position exists (recommendations should represent current target exposure)
        if 'Position' not in signals.columns:
            continue

        # Latest target position (regime) drives the recommendation
        last_pos = int(signals['Position'].iloc[-1])
        rec_pos = last_pos

        if persistence > 1:
            if len(signals) >= persistence:
                last_n_pos = signals['Position'].iloc[-persistence:]
                if not (last_n_pos == last_n_pos.iloc[0]).all():
                    rec_pos = 0
            else:
                rec_pos = 0

        rec = 'Buy' if rec_pos == 1 else ('Sell' if rec_pos == -1 else 'Hold')

        # Signal is an execution/event column (sparse). Keep it as extra info if present.
        last_sig = 0
        if 'Signal' in signals.columns and len(signals['Signal']) > 0:
            try:
                last_sig = int(signals['Signal'].iloc[-1])
            except Exception:
                last_sig = 0
        trade_now = bool(last_sig != 0)

        score = 0
        try:
            score = agent.returns_data.get(stock, {}).get(f"{agent.algorithm_name}_return", 0)
        except Exception:
            score = 0

        rows.append({
            'Stock': stock,
            'Recommendation': rec,
            'Position': rec_pos,
            'TradeNow': trade_now,
            'Signal': last_sig,
            'Score': score
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if min_score is not None:
        df = df[df['Score'] >= min_score]

    df = df.sort_values('Score', ascending=False).reset_index(drop=True)

    if top_n is not None:
        df = df.head(top_n)

    if save_path:
        df.to_csv(save_path, index=False)

    return df
