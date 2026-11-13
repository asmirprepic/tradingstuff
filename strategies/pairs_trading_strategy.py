"""Event-driven long/short pairs with a signed-share cash ledger."""
import numpy as np
import pandas as pd

from utilities.pairs_utils import select_pairs, spread_z


class PairsTradingStrategy:
    def __init__(self, data, formation_window=252, selection_interval=20,
                 minimum_correlation=.6, significance=.05, max_candidates=100,
                 entry_z=2., exit_z=.25, stop_z=4., max_holding_period=40,
                 max_pairs=5, gross_exposure=1., transaction_cost=.001,
                 annual_borrow_cost=.03, evaluation_start=None,
                 performance_benchmark=None, verbose=False, auto_generate=True,
                 correction='bonferroni_all', max_gross_exposure=None,
                 allocation_mode='active_equal'):
        if allocation_mode not in {'active_equal', 'active_beta', 'fixed_slots'}:
            raise ValueError('allocation_mode must be active_equal, active_beta or fixed_slots')
        self.allocation_mode = allocation_mode
        if correction not in {'bonferroni_all', 'bonferroni_tested', 'none'}:
            raise ValueError('correction must be bonferroni_all, bonferroni_tested or none')
        self.correction = correction
        if (not isinstance(data, pd.DataFrame) or data.empty
                or not isinstance(data.index, pd.DatetimeIndex)
                or not data.index.is_unique or not data.index.is_monotonic_increasing
                or not isinstance(data.columns, pd.MultiIndex) or data.columns.nlevels != 2
                or not data.columns.is_unique):
            raise ValueError('Expected sorted unique dates and (Stock, Field) columns')
        for name, value, lower in [('formation_window', formation_window, 40),
                ('selection_interval', selection_interval, 1), ('max_candidates', max_candidates, 1),
                ('max_pairs', max_pairs, 1), ('max_holding_period', max_holding_period, 1)]:
            if isinstance(value, bool) or not np.isfinite(value) or int(value) != value or value < lower:
                raise ValueError(f'{name} must be an integer >= {lower}')
        if not all(np.isfinite(v) for v in [minimum_correlation, significance, entry_z,
                exit_z, stop_z, gross_exposure, transaction_cost, annual_borrow_cost]):
            raise ValueError('Parameters must be finite')
        if not (0 <= exit_z < entry_z < stop_z and 0 <= minimum_correlation <= 1
                and 0 < significance <= 1 and 0 < gross_exposure <= 2
                and 0 <= transaction_cost < .1 and annual_borrow_cost >= 0):
            raise ValueError('Invalid thresholds, exposure or costs')
        self.data = data
        self.close = data.xs('Close', level=1, axis=1).astype(float)
        self.open = data.xs('Open', level=1, axis=1).reindex(columns=self.close.columns).astype(float)
        if len(self.close.columns) < 2 or len(data) < formation_window+2:
            raise ValueError('Need at least two stocks and formation_window + 2 rows')
        self.formation_window, self.selection_interval = int(formation_window), int(selection_interval)
        self.minimum_correlation, self.significance = minimum_correlation, significance
        self.max_candidates, self.max_pairs = int(max_candidates), int(max_pairs)
        self.entry_z, self.exit_z, self.stop_z = entry_z, exit_z, stop_z
        self.max_holding_period = int(max_holding_period)
        self.gross_exposure, self.transaction_cost = gross_exposure, transaction_cost
        self.max_gross_exposure = (gross_exposure*1.2 if max_gross_exposure is None
                                   else max_gross_exposure)
        if (not np.isfinite(self.max_gross_exposure)
                or self.max_gross_exposure < gross_exposure):
            raise ValueError('max_gross_exposure must be finite and >= gross_exposure')
        self.annual_borrow_cost = annual_borrow_cost
        self.performance_benchmark, self.verbose = performance_benchmark, verbose
        self.start_position = self.formation_window+1
        if evaluation_start is not None:
            pos = data.index.searchsorted(pd.Timestamp(evaluation_start))
            if pos < self.start_position or pos >= len(data):
                raise ValueError('evaluation_start must follow warm-up and be within data')
            self.start_position = int(pos)
        self.evaluation_start = data.index[self.start_position]
        self.strategy_name = 'PairsTrading'
        if auto_generate:
            self.run_all()

    @staticmethod
    def _value(trade, prices):
        p = prices.reindex(trade['Quantity'].index)
        if not np.isfinite(p).all() or p.le(0).any():
            raise ValueError(f"Missing execution/valuation price for held pair {trade['A']}/{trade['B']}")
        return trade['Quantity']*p

    def run_all(self):
        cash, previous_equity = 1., 1.
        active, pending_entries, pending_exits = [], [], {}
        candidates = pd.DataFrame()
        logs, trades, snapshots = [], [], {}
        diagnostics = []
        fills = []
        for t in range(self.start_position-1, len(self.data)):
            date = self.data.index[t]
            opening, closing = self.open.iloc[t], self.close.iloc[t]
            turnover, fees, borrow = 0., 0., 0.
            membership_changed = False
            # Charge elapsed calendar-day borrow against previous closing shorts.
            for trade in active:
                short = -self._value(trade, self.close.iloc[t-1]).clip(upper=0).sum()
                days = (date.date()-self.data.index[t-1].date()).days
                charge = short*self.annual_borrow_cost*days/365
                borrow += charge
                trade['BorrowCost'] += charge
            cash -= borrow
            for trade in active[:]:
                key = (trade['A'], trade['B'])
                if key not in pending_exits:
                    continue
                value = self._value(trade, opening)
                cost = value.abs().sum()*self.transaction_cost
                cash += value.sum()-cost
                turnover += value.abs().sum()
                fees += cost
                pnl_a = value.iloc[0]-trade['NetInvestedA']
                pnl_b = value.iloc[1]-trade['NetInvestedB']
                pnl = pnl_a+pnl_b-trade['EntryCost']-trade['ResizeCost']-cost-trade['BorrowCost']
                for stock, qty in trade['Quantity'].items():
                    fills.append(dict(Date=date, TradeID=trade['TradeID'], Stock=stock,
                                      Quantity=-qty, Price=opening[stock], Reason='EXIT',
                                      Cost=abs(qty*opening[stock])*self.transaction_cost))
                trades.append({k: v for k, v in trade.items() if k != 'Quantity'} | {
                    'ExitDate': date, 'ExitReason': pending_exits[key], 'PnL': pnl,
                    'ExitPriceA': opening[trade['A']], 'ExitPriceB': opening[trade['B']],
                    'GrossPnLA': pnl_a, 'GrossPnLB': pnl_b,
                    'LongGrossPnL': pnl_a if trade['QuantityA'] > 0 else pnl_b,
                    'ShortGrossPnL': pnl_a if trade['QuantityA'] < 0 else pnl_b,
                    'PairReturn_%': 100*pnl/trade['EntryGross'], 'ExitCost': cost})
                active.remove(trade)
                membership_changed = True
            pending_exits = {}
            equity_open = cash + sum(self._value(p, opening).sum() for p in active)
            if equity_open <= 0:
                raise ValueError('Portfolio insolvent at open; margin/liquidation not modeled')
            used = {s for p in active for s in [p['A'], p['B']]}
            gross = sum(self._value(p, opening).abs().sum() for p in active)
            for pair in pending_entries:
                if len(active) >= self.max_pairs or used.intersection([pair['A'], pair['B']]):
                    continue
                prices = opening.reindex([pair['A'], pair['B']])
                if not np.isfinite(prices).all() or prices.le(0).any():
                    continue  # Cancel both legs atomically; never partially fill.
                budget = min(equity_open*self.gross_exposure/self.max_pairs,
                             (equity_open*self.gross_exposure-gross)/
                             (1+self.gross_exposure*self.transaction_cost))
                if self.allocation_mode in {'active_equal', 'active_beta'}:
                    budget = 0.  # Size all accepted pairs together after membership is known.
                elif budget <= 1e-12:
                    continue
                side = -np.sign(pair['EntryZ'])
                dollars = pd.Series([side, -side*pair['Beta']], index=prices.index)
                dollars *= budget/(1+pair['Beta'])
                cost = budget*self.transaction_cost
                quantity = dollars/prices
                cash -= dollars.sum()+cost
                fees += cost
                turnover += budget
                equity_open -= cost
                gross += budget
                trade = dict(pair, Quantity=quantity, EntryDate=date, EntryPosition=t,
                             TradeID=len(trades)+len(active),
                             NetInvestedA=float(dollars.iloc[0]), NetInvestedB=float(dollars.iloc[1]),
                             ResizeCost=0.,
                             LongStock=pair['A'] if side > 0 else pair['B'],
                             ShortStock=pair['A'] if side < 0 else pair['B'],
                             QuantityA=float(quantity.iloc[0]), QuantityB=float(quantity.iloc[1]),
                             EntryPriceA=float(prices.iloc[0]), EntryPriceB=float(prices.iloc[1]),
                             EntryValue=float(dollars.sum()), EntryGross=budget,
                             EntryCost=cost, BorrowCost=0.)
                active.append(trade)
                membership_changed = True
                if self.allocation_mode == 'fixed_slots':
                    for stock, qty in quantity.items():
                        fills.append(dict(Date=date, TradeID=trade['TradeID'], Stock=stock,
                                          Quantity=qty, Price=opening[stock], Reason='ENTRY',
                                          Cost=abs(qty*opening[stock])*self.transaction_cost))
                used.update(prices.index)
            pending_entries = []
            if self.allocation_mode in {'active_equal', 'active_beta'} and membership_changed and active:
                cash, traded, resizing_fees = self._resize_active(active, opening, cash, date, fills)
                turnover += traded
                fees += resizing_fees
            values = [self._value(p, closing) for p in active]
            equity = cash+sum(v.sum() for v in values)
            if equity <= 0:
                raise ValueError('Portfolio insolvent at close')
            long_value = sum(v.clip(lower=0).sum() for v in values)
            short_value = -sum(v.clip(upper=0).sum() for v in values)
            if t >= self.start_position:
                logs.append(dict(Date=date, Equity=equity, PortfolioReturn=equity/previous_equity-1,
                                 LongExposure=long_value/equity, ShortExposure=short_value/equity,
                                 GrossExposure=(long_value+short_value)/equity,
                                 NetExposure=(long_value-short_value)/equity, N_Pairs=len(active),
                                 Turnover=turnover/previous_equity, TradingCost=fees, BorrowCost=borrow))
            previous_equity = equity
            # At close t, estimation ends at t-1; signals may use close t.
            if (t-(self.start_position-1)) % self.selection_interval == 0:
                candidates = select_pairs(self.close.iloc[t-self.formation_window:t],
                    self.minimum_correlation, self.significance, self.max_candidates,
                    correction=self.correction)
                snapshots[date] = candidates.copy()
                info = candidates.attrs.get('diagnostics', {})
                diagnostics.append(dict(Date=date, **info))
                if self.verbose:
                    print(f'{date}: {len(candidates)} eligible pairs | '
                          f"valid stocks={info.get('ValidStocks', '?')} | "
                          f"correlation passed={info.get('CorrelationPassed', '?')} | "
                          f"tested={info.get('TestsAttempted', '?')} | "
                          f"min p={info.get('MinimumPValue', float('nan')):.3g} | "
                          f"raw significant={info.get('RawSignificant', '?')} | "
                          f"correction={self.correction}", flush=True)
            for trade in active:
                z = spread_z(trade, closing)
                reason = None
                if abs(z) >= self.stop_z:
                    reason = 'STOP'
                elif z*trade['EntryZ'] <= 0 or abs(z) <= self.exit_z:
                    reason = 'CONVERGENCE'
                elif t-trade['EntryPosition']+1 >= self.max_holding_period:
                    reason = 'TIME_LIMIT'
                if reason:
                    pending_exits[(trade['A'], trade['B'])] = reason
            risk_breach = self._queue_exposure_exits(active, closing, equity, pending_exits)
            used = {s for p in active for s in [p['A'], p['B']]}
            for pair in candidates.to_dict('records'):
                if risk_breach:
                    break  # Do not replace exposure on the same risk-reduction cycle.
                if used.intersection([pair['A'], pair['B']]):
                    continue
                z = spread_z(pair, closing)
                if np.isfinite(z) and self.entry_z <= abs(z) < self.stop_z:
                    pending_entries.append(dict(pair, EntryZ=z, DecisionDate=date))
        self.selection_log = pd.DataFrame(logs).set_index('Date')
        self.portfolio_returns = self.selection_log.PortfolioReturn
        self.equity_curve = self.selection_log.Equity
        self.trade_log = pd.DataFrame(trades)
        self.fill_log = pd.DataFrame(fills)
        self.candidate_history = snapshots
        self.selection_diagnostics = pd.DataFrame(diagnostics).set_index('Date')
        self.open_pairs = pd.DataFrame([{k: v for k, v in p.items() if k != 'Quantity'} for p in active])
        self.pending_entries = pd.DataFrame(pending_entries)
        self.pending_exits = pending_exits
        return self

    def _resize_active(self, active, prices, cash, date, fills):
        current = [self._value(p, prices) for p in active]
        equity = cash + sum(v.sum() for v in current)
        signs = []
        for pair, value in zip(active, current):
            beta = pair['Beta'] if self.allocation_mode == 'active_beta' else 1.
            side = -np.sign(pair['EntryZ'])
            signs.append(pd.Series([side, -side*beta], index=value.index)/(1+beta))
        # Solve gross = target * post-cost equity. Fees depend on resizing turnover.
        def cost(gross):
            return self.transaction_cost*sum(
                (s*gross/len(active)-v).abs().sum() for s,v in zip(signs,current))
        if equity-cost(0.) <= 0:
            raise ValueError('Resizing costs consume portfolio equity')
        low, high = 0., self.gross_exposure*equity
        for _ in range(60):
            mid = (low+high)/2
            if mid > self.gross_exposure*(equity-cost(mid)):
                high = mid
            else:
                low = mid
        gross = (low+high)/2
        turnover, fees = 0., 0.
        for trade, old, sign in zip(active, current, signs):
            target = sign*gross/len(active)
            delta = target-old
            charge = float(delta.abs().sum()*self.transaction_cost)
            is_entry = trade['EntryGross'] == 0
            cash -= float(delta.sum())+charge
            turnover += float(delta.abs().sum())
            fees += charge
            trade['NetInvestedA'] += float(delta.iloc[0])
            trade['NetInvestedB'] += float(delta.iloc[1])
            trade['EntryCost' if is_entry else 'ResizeCost'] += charge
            trade['Quantity'] = target/prices.reindex(target.index)
            if is_entry:
                trade['EntryGross'] = float(target.abs().sum())
                trade['QuantityA'], trade['QuantityB'] = trade['Quantity'].tolist()
            for stock, dollars in delta.items():
                if abs(dollars) > 1e-15:
                    fills.append(dict(Date=date, TradeID=trade['TradeID'], Stock=stock,
                                      Quantity=dollars/prices[stock], Price=prices[stock],
                                      Reason='ENTRY' if is_entry else 'RESIZE',
                                      Cost=abs(dollars)*self.transaction_cost))
        return cash, turnover, fees

    def _queue_exposure_exits(self, active, prices, equity, pending_exits):
        """At a ceiling breach, queue largest whole pairs until back at target.

        Estimates use current close and allow for closing fees. Actual next-open
        prices may gap, so this is a risk policy, not a guaranteed exposure cap.
        """
        sizes = [(p, float(self._value(p, prices).abs().sum())) for p in active]
        remaining = sum(size for _, size in sizes)
        if remaining <= self.max_gross_exposure*equity + 1e-12:
            return False
        projected_equity = equity
        for pair, size in sizes:
            if (pair['A'], pair['B']) in pending_exits:
                remaining -= size
                projected_equity -= size*self.transaction_cost
        for pair, size in sorted(sizes, key=lambda item: (-item[1], item[0]['A'], item[0]['B'])):
            if remaining <= self.gross_exposure*projected_equity + 1e-12:
                break
            key = (pair['A'], pair['B'])
            if key not in pending_exits:
                pending_exits[key] = 'EXPOSURE_LIMIT'
                remaining -= size
                projected_equity -= size*self.transaction_cost
        return True

    def evaluate_performance(self):
        from strategies.base_top_n_strategy import BaseTopNStrategy
        def metrics(returns):
            result = BaseTopNStrategy._performance_metrics(returns)
            eq = (1+returns).cumprod()
            result['Max_Drawdown_%'] = 100*(eq/eq.cummax().clip(lower=1)-1).min()
            return result
        rows = [{'Portfolio': self.strategy_name, **metrics(self.portfolio_returns)}]
        if self.performance_benchmark is not None:
            b = self.performance_benchmark
            baseline = self.data.index[self.start_position-1]
            prices = b.prices_for_period(baseline, self.data.index[-1])
            dates = self.data.index[self.start_position-1:].tz_localize(None).normalize()
            prices = prices.reindex(dates)
            if prices.isna().any():
                raise ValueError('Benchmark prices missing on evaluation dates')
            returns = prices.pct_change(fill_method=None).iloc[1:]
            rows.append({'Portfolio': b.name, **metrics(returns)})
        result = pd.DataFrame(rows)
        if len(rows) > 1:
            result['Outperformance_%'] = result['Total_Return_%']-result.iloc[1]['Total_Return_%']
        return result.round(4)
