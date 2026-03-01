import pandas as pd


def evaluations_from_agent(agent, persistence=1, min_score=None, top_n=None, save_path=None):
    """Generate ranked per-stock recommendations from a trading agent.

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
          - AsOf (timestamp of the latest usable row)
          - Action (BUY/SELL/HOLD; from latest Signal when available)
          - Recommendation (Buy/Sell/Hold; based on latest stable Position)
          - Position (target exposure: -1/0/1)
          - TradeNow (bool; True if last Signal != 0)
          - Signal (last trade event: -1/0/1 if present else 0)
          - Score (ranking score; prefers agent.score_now(stock) when available)
          - TotalReturnScore (historical strategy return score, when available)
        sorted by Score desc.
    """

    rows = []

    for stock, signals in agent.signal_data.items():
        if signals is None or signals.empty:
            continue

        # Ensure Position exists (recommendations should represent current target exposure)
        if 'Position' not in signals.columns:
            continue

        # Use agent-provided "latest usable row" logic when possible (drops NaN returns etc.)
        latest = None
        if hasattr(agent, "latest_row"):
            try:
                latest = agent.latest_row(stock)
            except Exception:
                latest = None
        if latest is None:
            latest = signals.iloc[-1]

        as_of = latest.name

        # Latest target position (regime) drives the recommendation
        try:
            last_pos = int(latest.get('Position', 0))
        except Exception:
            last_pos = 0

        rec_pos = last_pos
        if persistence and int(persistence) > 1:
            p = int(persistence)
            ref = signals
            if "return" in ref.columns:
                ref = ref.dropna(subset=["return"])
            if len(ref) >= p:
                last_n_pos = ref['Position'].iloc[-p:]
                if not (last_n_pos == last_n_pos.iloc[0]).all():
                    rec_pos = 0
            else:
                rec_pos = 0

        rec = 'Buy' if rec_pos == 1 else ('Sell' if rec_pos == -1 else 'Hold')

        # Signal is an execution/event column (sparse). Keep it as extra info if present.
        last_sig = 0
        if 'Signal' in signals.columns and len(signals['Signal']) > 0:
            try:
                last_sig = int(latest.get('Signal', 0))
            except Exception:
                last_sig = 0
        trade_now = bool(last_sig != 0)

        # Ranking score: prefer "today's" score if agent supports it, else fall back to historical return score.
        score = float("nan")
        if hasattr(agent, "score_now"):
            try:
                score = float(agent.score_now(stock))
            except Exception:
                score = float("nan")

        total_return_score = None
        try:
            total_return_score = agent.returns_data.get(stock, {}).get(f"{agent.algorithm_name}_return", None)
        except Exception:
            total_return_score = None

        if pd.isna(score):
            try:
                score = float(total_return_score) if total_return_score is not None else float("nan")
            except Exception:
                score = float("nan")

        action = None
        if hasattr(agent, "action_now"):
            try:
                action = agent.action_now(stock).get("Action")
            except Exception:
                action = None
        if not action:
            action = "BUY" if last_sig == 1 else ("SELL" if last_sig == -1 else "HOLD")

        latest_features = {}
        extra_cols = ["SignalStrength", "Momentum"]
        score_col = getattr(agent, "score_column", None)
        if score_col and score_col not in extra_cols:
            extra_cols.append(score_col)

        for col in extra_cols:
            if col in signals.columns:
                try:
                    latest_features[col] = float(latest.get(col))
                except Exception:
                    latest_features[col] = None

        rows.append({
            'Stock': stock,
            'AsOf': as_of,
            'Action': action,
            'Recommendation': rec,
            'Position': rec_pos,
            'TradeNow': trade_now,
            'Signal': last_sig,
            'Score': score,
            'TotalReturnScore': total_return_score,
            **latest_features,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if min_score is not None:
        df = df[df['Score'].fillna(float("-inf")) >= min_score]

    df = df.sort_values('Score', ascending=False, na_position='last').reset_index(drop=True)

    if top_n is not None:
        df = df.head(top_n)

    if save_path:
        df.to_csv(save_path, index=False)

    return df
