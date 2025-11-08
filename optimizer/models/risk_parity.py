import io, base64
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# -------------------------
# Helpers
# -------------------------
def fig_to_data_uri(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def correl_dist_from_cov(cov: pd.DataFrame) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov.values / denom
    corr = np.nan_to_num(corr, nan=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    return dist

# -------------------------
# HRP & HERC implementations (operate on covariance DataFrame)
# -------------------------
def hrp_weights_from_cov(cov: pd.DataFrame) -> pd.Series:
    tickers = cov.index.tolist()
    if len(tickers) == 1:
        return pd.Series([1.0], index=tickers)
    dist = correl_dist_from_cov(cov)
    dist_cond = squareform(dist, checks=False)
    link = hierarchy.linkage(dist_cond, method="single")
    leaves = hierarchy.dendrogram(link, no_plot=True)['leaves']
    sorted_tickers = [tickers[int(i)] for i in leaves]  # <- FIXED: explicit int cast
    cov_sorted = cov.reindex(index=sorted_tickers, columns=sorted_tickers)

    weights = pd.Series(1.0, index=sorted_tickers, dtype=float)
    clusters = [sorted_tickers]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) == 1:
            continue
        split = int(len(cluster) // 2)  # <- FIXED: explicit int cast
        left = cluster[:split]
        right = cluster[split:]
        left_w = np.ones(len(left)) / len(left)
        right_w = np.ones(len(right)) / len(right)
        left_var = float(left_w @ cov_sorted.loc[left, left].values @ left_w)
        right_var = float(right_w @ cov_sorted.loc[right, right].values @ right_w)
        if (left_var + right_var) == 0:
            alpha = 0.5
        else:
            alpha = 1.0 - left_var / (left_var + right_var)
        weights[left] *= alpha
        weights[right] *= (1.0 - alpha)
        clusters.append(left)
        clusters.append(right)

    weights = weights / weights.sum()
    return weights.reindex(index=tickers).fillna(0.0)

def herc_weights_from_cov(cov: pd.DataFrame) -> pd.Series:
    """Simple HERC-like heuristic using inverse-vol at subcluster level, then normalize."""
    tickers = cov.index.tolist()
    if len(tickers) == 1:
        return pd.Series([1.0], index=tickers)
    dist = correl_dist_from_cov(cov)
    link = hierarchy.linkage(squareform(dist, checks=False), method="single")
    leaves = hierarchy.dendrogram(link, no_plot=True)['leaves']
    sorted_tickers = [tickers[int(i)] for i in leaves]  # <- FIXED: explicit int cast

    weights = pd.Series(0.0, index=sorted_tickers, dtype=float)
    clusters = [sorted_tickers]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) == 1:
            weights[cluster[0]] = 1.0
            continue
        split = int(len(cluster) // 2)  # <- FIXED: explicit int cast
        left = cluster[:split]
        right = cluster[split:]
        subcov_l = cov.loc[left, left]
        subcov_r = cov.loc[right, right]
        invvol_l = 1.0 / np.sqrt(np.diag(subcov_l))
        invvol_r = 1.0 / np.sqrt(np.diag(subcov_r))
        if invvol_l.sum() > 0:
            wl = invvol_l / invvol_l.sum()
        else:
            wl = np.ones(len(left)) / len(left)
        if invvol_r.sum() > 0:
            wr = invvol_r / invvol_r.sum()
        else:
            wr = np.ones(len(right)) / len(right)
        for i, t in enumerate(left):
            weights[t] = wl[i]
        for i, t in enumerate(right):
            weights[t] = wr[i]
        clusters.append(left)
        clusters.append(right)
    weights = weights / weights.sum()
    return weights.reindex(index=tickers).fillna(0.0)

# -------------------------
# Simulation: rolling 30-day cov + 5% stop-loss rebalancing
# -------------------------
def simulate_model_with_stoploss(prices: pd.DataFrame, model: str, window=30, stop_loss: float=0.05
                                 ) -> Tuple[pd.Series, pd.DataFrame, int]:
    prices = prices.sort_index()
    returns = prices.pct_change().dropna()
    if len(returns) < window + 1:
        raise ValueError(f"Need at least window+1 returns ({window+1}), got {len(returns)}")

    tickers = returns.columns.tolist()
    sim_dates = returns.index[window:]
    growth = pd.Series(index=sim_dates, dtype=float)
    weight_history = pd.DataFrame(index=sim_dates, columns=tickers, dtype=float)

    current_weights = None
    value = 1.0
    peak_value = 1.0
    rebalance_count = 0

    for i in range(window, len(returns)):
        date = returns.index[i]
        # compute weights using covariance of the prior 'window' returns (no look-ahead)
        if current_weights is None:
            cov_window = returns.iloc[i-window:i].cov()
            if model == "equal":
                current_weights = pd.Series(1.0 / len(tickers), index=tickers)
            elif model == "hrp":
                current_weights = hrp_weights_from_cov(cov_window)
            elif model == "herc":
                current_weights = herc_weights_from_cov(cov_window)
            else:
                raise ValueError("model must be 'equal', 'hrp' or 'herc'")

        # record weights (weights in effect that day)
        weight_history.loc[date] = current_weights.values

        # apply day's return
        daily_ret = float((current_weights * returns.iloc[i]).sum())
        value = value * (1.0 + daily_ret)
        growth.loc[date] = value

        # update peak and check stop-loss
        if value > peak_value:
            peak_value = value

        if value <= peak_value * (1.0 - stop_loss):
            # recompute weights using window ending today (inclusive) to avoid look-ahead
            start = max(0, i - window + 1)
            cov_window = returns.iloc[start:i+1].cov()
            if model == "equal":
                current_weights = pd.Series(1.0 / len(tickers), index=tickers)
            elif model == "hrp":
                current_weights = hrp_weights_from_cov(cov_window)
            elif model == "herc":
                current_weights = herc_weights_from_cov(cov_window)
            rebalance_count += 1
            peak_value = value  # reset peak after rebalance

    return growth, weight_history.fillna(method='ffill').fillna(0.0), rebalance_count

# -------------------------
# Metrics from growth series
# -------------------------
def metrics_from_growth(growth: pd.Series) -> Dict[str, float]:
    if growth.isna().all() or len(growth.dropna()) < 2:
        return {'expected_return': 0.0, 'volatility': 0.0, 'sharpe_ratio': 0.0}
    daily_rets = growth.pct_change().dropna()
    ann_ret = float(daily_rets.mean() * 252)
    ann_vol = float(daily_rets.std() * (252 ** 0.5))
    sharpe = float(ann_ret / ann_vol) if ann_vol != 0 else 0.0
    return {'expected_return': ann_ret, 'volatility': ann_vol, 'sharpe_ratio': sharpe}

# -------------------------
# Plotting helpers
# -------------------------
def plot_growth_comparison(growth_dict: Dict[str, pd.Series]) -> str:
    fig, ax = plt.subplots(figsize=(9,5))
    for label, series in growth_dict.items():
        ax.plot(series.index, series.values, label=label)
    ax.set_title("Portfolio Growth (stop-loss rebalancing)")
    ax.set_ylabel("Portfolio Value (start=1)")
    ax.legend()
    return fig_to_data_uri(fig)

def plot_weight_evolution(weight_history: pd.DataFrame, title: str="Weight Evolution") -> str:
    wf = weight_history.astype(float)
    fig, ax = plt.subplots(figsize=(10,6))
    wf.plot.area(ax=ax, stacked=True)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    return fig_to_data_uri(fig)

def plot_dendrogram_from_cov(cov: pd.DataFrame) -> str:
    tickers = cov.index.tolist()
    if len(tickers) == 1:
        fig, ax = plt.subplots(figsize=(4,2))
        ax.text(0.5, 0.5, tickers[0], ha='center', va='center')
        return fig_to_data_uri(fig)
    dist = correl_dist_from_cov(cov)
    link = hierarchy.linkage(squareform(dist, checks=False), method="single")
    fig, ax = plt.subplots(figsize=(8,4))
    hierarchy.dendrogram(link, labels=tickers, ax=ax, leaf_rotation=45)
    ax.set_title("Dendrogram (last window)")
    fig.tight_layout()
    return fig_to_data_uri(fig)

def plot_quasi_diag_heatmap(cov: pd.DataFrame) -> str:
    tickers = cov.index.tolist()
    if len(tickers) == 1:
        fig, ax = plt.subplots(figsize=(3,3))
        ax.text(0.5,0.5, tickers[0], ha='center', va='center')
        return fig_to_data_uri(fig)
    dist = correl_dist_from_cov(cov)
    link = hierarchy.linkage(squareform(dist, checks=False), method="single")
    leaves = hierarchy.dendrogram(link, no_plot=True)['leaves']
    ordered = [tickers[int(i)] for i in leaves]  # <- FIXED: explicit int cast
    cov_q = cov.reindex(index=ordered, columns=ordered)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cov_q.values, aspect='auto')
    ax.set_xticks(range(len(ordered)))
    ax.set_yticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=90, fontsize=8)
    ax.set_yticklabels(ordered, fontsize=8)
    ax.set_title("Quasi-diagonalized Covariance (last window)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    return fig_to_data_uri(fig)

# -------------------------
# Main entry: risk_parity
# -------------------------
def risk_parity(prices: pd.DataFrame, window: int = 30, stop_loss: float = 0.05) -> str:
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("prices must be a pandas DataFrame")

    prices = prices.sort_index()
    # simulate each model
    growth_eq, weights_eq, reb_eq = simulate_model_with_stoploss(prices, "equal", window, stop_loss)
    growth_hrp, weights_hrp, reb_hrp = simulate_model_with_stoploss(prices, "hrp", window, stop_loss)
    growth_herc, weights_herc, reb_herc = simulate_model_with_stoploss(prices, "herc", window, stop_loss)

    metrics_eq = metrics_from_growth(growth_eq)
    metrics_hrp = metrics_from_growth(growth_hrp)
    metrics_herc = metrics_from_growth(growth_herc)

    # Portfolio growth plot: HRP and Equal only
    growth_img = plot_growth_comparison({
        "Equal Weight": growth_eq,
        "HRP (rolling window)": growth_hrp
    })

    # weight evolution plots (HRP & HERC)
    weight_evol_hrp = plot_weight_evolution(weights_hrp, title="HRP Weight Evolution")
    weight_evol_herc = plot_weight_evolution(weights_herc, title="HERC Weight Evolution")

    # last-window cov for dendrogram/heatmap
    returns = prices.pct_change().dropna()
    if len(returns) >= window:
        cov_last = returns.iloc[-window:].cov()
    else:
        cov_last = returns.cov()

    dendro_img = plot_dendrogram_from_cov(cov_last)
    heatmap_img = plot_quasi_diag_heatmap(cov_last)

    # final weights (weights in effect at the simulation end)
    final_eq = weights_eq.iloc[-1].round(6) if not weights_eq.empty else pd.Series()
    final_hrp = weights_hrp.iloc[-1].round(6) if not weights_hrp.empty else pd.Series()
    tickers = prices.columns.tolist()

    # build weights table rows (only Equal and HRP shown)
    rows = ""
    for t in tickers:
        we = final_eq.get(t, 0.0)
        wh = final_hrp.get(t, 0.0)
        rows += f"""
        <tr>
            <td class="px-4 py-2">{t}</td>
            <td class="px-4 py-2 text-right">{we:.2%}</td>
            <td class="px-4 py-2 text-right">{wh:.2%}</td>
        </tr>
        """

    weights_table = f"""
    <table class="min-w-full table-auto divide-y divide-gray-200 text-sm text-gray-800">
      <thead class="bg-gray-50"><tr>
        <th class="px-4 py-2 text-left">Ticker</th>
        <th class="px-4 py-2 text-right">Equal (final)</th>
        <th class="px-4 py-2 text-right">HRP (final)</th>
      </tr></thead>
      <tbody class="bg-white divide-y divide-gray-100">{rows}</tbody>
    </table>
    """

    metrics_html = f"""
    <table class="min-w-full divide-y divide-gray-200 text-sm text-gray-800">
      <thead class="bg-gray-50">
        <tr><th class="px-4 py-2 text-left">Model</th>
            <th class="px-4 py-2 text-right">Expected Return</th>
            <th class="px-4 py-2 text-right">Volatility</th>
            <th class="px-4 py-2 text-right">Sharpe</th></tr>
      </thead>
      <tbody class="bg-white divide-y divide-gray-100">
        <tr><td class="px-4 py-2">Equal</td><td class="px-4 py-2 text-right">{metrics_eq['expected_return']:.2%}</td><td class="px-4 py-2 text-right">{metrics_eq['volatility']:.2%}</td><td class="px-4 py-2 text-right">{metrics_eq['sharpe_ratio']:.2f}</td></tr>
        <tr><td class="px-4 py-2">HRP</td><td class="px-4 py-2 text-right">{metrics_hrp['expected_return']:.2%}</td><td class="px-4 py-2 text-right">{metrics_hrp['volatility']:.2%}</td><td class="px-4 py-2 text-right">{metrics_hrp['sharpe_ratio']:.2f}</td></tr>
        <tr><td class="px-4 py-2">HERC</td><td class="px-4 py-2 text-right">{metrics_herc['expected_return']:.2%}</td><td class="px-4 py-2 text-right">{metrics_herc['volatility']:.2%}</td><td class="px-4 py-2 text-right">{metrics_herc['sharpe_ratio']:.2f}</td></tr>
      </tbody>
    </table>
    """

    html_output = f"""
    <div class="flex flex-col space-y-6 mt-6">

      <!-- Final weights -->
      <div class="overflow-x-auto rounded-lg shadow">
        <h2 class="text-lg font-semibold mb-3">Final Weights (weights in effect at simulation end)</h2>
        {weights_table}
      </div>

      <!-- Summary -->
      <div class="bg-gray-100 p-6 rounded shadow">
        <h2 class="text-lg font-semibold mb-3">Portfolio Summary (stop-loss adjusted)</h2>
        {metrics_html}
        <p class="text-sm text-gray-600 mt-2">Rebalances (Equal / HRP / HERC): {reb_eq} / {reb_hrp} / {reb_herc}</p>
      </div>

      <!-- Growth -->
      <div>
        <h2 class="text-lg font-semibold mb-3">Portfolio Growth (Equal + HRP Only)</h2>
        <img src="{growth_img}" class="w-full rounded border border-gray-300 shadow" />
      </div>

      <!-- Weight evolution -->
      <div>
        <h2 class="text-lg font-semibold mb-3">HRP Weight Evolution</h2>
        <img src="{weight_evol_hrp}" class="w-full rounded border border-gray-300 shadow" />

      <!-- Dendrogram and heatmap (last window) -->
      <div>
        <h2 class="text-lg font-semibold mb-3">Dendrogram (last window)</h2>
        <img src="{dendro_img}" class="w-full rounded border border-gray-300 shadow" />
      </div>
      <div>
        <h2 class="text-lg font-semibold mb-3">Quasi-diagonal Covariance (last window)</h2>
        <img src="{heatmap_img}" class="w-full rounded border border-gray-300 shadow" />
      </div>

    </div>
    """
    return html_output





