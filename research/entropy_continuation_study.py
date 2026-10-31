import numpy as np
import pandas as pd
import statsmodels.api as sm

from agents.technical.cross_sectional_entropy_agent import CrossSectionalEntropyAgent


def entropy_continuation_summary(event_history, quantiles=5):
    """Summarize forward leader excess returns by ex-post entropy regime."""
    events = pd.DataFrame(event_history).copy()
    required = {"UpsideEntropy", "EntropyChange", "LeaderExcessReturn", "HorizonWeeks"}
    if events.empty or not required.issubset(events.columns):
        return pd.DataFrame()
    events = events.dropna(subset=list(required))
    if events.empty:
        return pd.DataFrame()
    quantile_count = min(int(quantiles), len(events))
    try:
        ranked_entropy = events["UpsideEntropy"].rank(method="first")
        events["EntropyRegime"] = pd.qcut(
            ranked_entropy,
            quantile_count,
            labels=[f"Q{index + 1}" for index in range(quantile_count)],
        )
    except ValueError:
        events["EntropyRegime"] = "ALL"
    events["EntropyDirection"] = np.where(
        events["EntropyChange"].lt(0.0), "FALLING", "RISING"
    )
    rows = []
    for keys, group in events.groupby(
        ["EntropyRegime", "EntropyDirection", "HorizonWeeks"], observed=True
    ):
        values = group["LeaderExcessReturn"].to_numpy(dtype=float)
        hac_t = np.nan
        if len(values) > 1:
            model = sm.OLS(values, np.ones((len(values), 1))).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": max(1, int(keys[2]))},
            )
            hac_t = float(model.tvalues[0])
        rows.append({
            "EntropyRegime": keys[0],
            "EntropyDirection": keys[1],
            "HorizonWeeks": keys[2],
            "LeaderExcessReturn": float(values.mean()),
            "HitRate": float(np.mean(values > 0.0)),
            "Samples": len(values),
            "HAC_t": hac_t,
        })
    return pd.DataFrame(rows).sort_values(
        ["EntropyRegime", "EntropyDirection", "HorizonWeeks"]
    ).reset_index(drop=True)


def run_entropy_continuation_study(data, **agent_parameters):
    agent = CrossSectionalEntropyAgent(data, **agent_parameters)
    return agent, entropy_continuation_summary(agent.event_history)
