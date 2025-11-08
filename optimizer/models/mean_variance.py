# optimizer/models/mean_variance.py

from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')


def mean_variance_optimization(price_df):
    returns_df = price_df.pct_change().dropna()

    mu = returns_df.mean() * 252
    cov = returns_df.cov() * 252
    tickers = returns_df.columns.tolist()
    n_assets = len(tickers)

    num_portfolios = 100000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(tickers)))
        weights_record.append(weights)
        port_return = np.dot(weights, mu)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe_ratio = port_return / port_volatility

        results[0, i] = port_return
        results[1, i] = port_volatility
        results[2, i] = sharpe_ratio

    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]
    optimal_return = results[0, max_sharpe_idx]
    optimal_volatility = results[1, max_sharpe_idx]
    optimal_sharpe = results[2, max_sharpe_idx]

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

    def get_optimized_portfolio(target_return):
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mu) - target_return}
        )
        bounds = tuple((0, 1) for _ in range(n_assets))
        result = minimize(portfolio_volatility,
                          x0=np.ones(n_assets)/n_assets,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
        return result

    target_returns = np.linspace(results[0].min(), results[0].max(), 100)
    frontier_volatilities = []

    for r in target_returns:
        res = get_optimized_portfolio(r)
        if res.success:
            frontier_volatilities.append(portfolio_volatility(res.x))
        else:
            frontier_volatilities.append(np.nan)

    fig, ax = plt.subplots()
    sc = ax.scatter(results[1], results[0], c=results[2],
                    cmap='viridis', alpha=0.6, label="Simulated Portfolios")
    ax.scatter(optimal_volatility, optimal_return, c='red',
               marker='*', s=150, label='Max Sharpe')

    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.set_title('Efficient Frontier')
    ax.legend()
    fig.colorbar(sc, label='Sharpe Ratio')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    image_png = buffer.getvalue()
    plot_url = base64.b64encode(image_png).decode('utf-8')

    # Format weights into DataFrame
    weights_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight': [f"{w:.2%}" for w in optimal_weights]
    })

    # Convert DataFrame to HTML table with Tailwind classes
    # weights_html = weights_df.to_html(
    #     index=False,
    #     classes="min-w-full divide-y divide-gray-200 text-sm text-gray-700"
    # )
    weights_html = f"""
    <table class="min-w-full table-auto divide-y divide-gray-200 text-sm text-gray-800">
        <thead class="bg-gray-50">
            <tr>
                <th class="px-4 py-2 text-left font-semibold">Ticker</th>
                <th class="px-4 py-2 text-left font-semibold">Weight</th>
            </tr>
        </thead>
        <tbody class="bg-white divide-y divide-gray-100">
            {''.join(f"<tr><td class='px-4 py-2 text-left'>{ticker}</td><td class='px-4 py-2 text-left'>{weight}</td></tr>" for ticker, weight in zip(tickers, [f'{w:.2%}' for w in optimal_weights]))}
        </tbody>
    </table>
    """

    # Full HTML output string
    html_output = f"""
    <div class="flex flex-col space-y-6 mt-6">

        <!-- Optimal Weights Table -->
        <div class="overflow-x-auto rounded-lg shadow">
            <h2 class="text-lg font-semibold mb-3">Optimal Weights</h2>
            <div class="bg-white overflow-hidden">
                {weights_html}
            </div>
        </div>

        <!-- Portfolio Summary -->
        <div class="bg-gray-100 p-6 rounded shadow">
            <h2 class="text-lg font-semibold mb-3">Portfolio Summary</h2>
            <ul class="list-disc pl-6 text-gray-800">
                <li><strong>Expected Return:</strong> {optimal_return:.2%}</li>
                <li><strong>Volatility:</strong> {optimal_volatility:.2%}</li>
                <li><strong>Sharpe Ratio:</strong> {optimal_sharpe:.2f}</li>
            </ul>
        </div>

        <!-- Efficient Frontier Plot -->
        <div class="rounded-lg">
            <h2 class="text-lg font-semibold mb-3">Efficient Frontier</h2>
            <img src="data:image/png;base64,{plot_url}" 
                alt="Efficient Frontier" 
                class="w-full rounded border border-gray-300 shadow" />
        </div>

    </div>
    """
    return html_output

    # summary_html = f"""
    #     <p><strong>Expected Return:</strong> {optimal_return:.2%}</p>
    #     <p><strong>Volatility:</strong> {optimal_volatility:.2%}</p>
    #     <p><strong>Sharpe Ratio:</strong> {optimal_sharpe:.2f}</p>
    # """

    # return weights_html + summary_html + img_html


# def mean_variance_optimization(returns_df):
#     returns_df = returns_df.pct_change().dropna()
#     # 1. Basic stats
#     mu = returns_df.mean() * 252  # Annualized mean
#     cov = returns_df.cov() * 252  # Annualized covariance
#     tickers = returns_df.columns.tolist()

#     # 2. Random portfolios for efficient frontier
#     num_portfolios = 5000
#     results = np.zeros((3, num_portfolios))
#     weights_record = []

#     for i in range(num_portfolios):
#         weights = np.random.dirichlet(np.ones(len(tickers)))
#         weights_record.append(weights)
#         port_return = np.dot(weights, mu)
#         port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
#         sharpe_ratio = port_return / port_volatility

#         results[0, i] = port_return
#         results[1, i] = port_volatility
#         results[2, i] = sharpe_ratio

#     # 3. Locate optimal portfolio (max Sharpe)
#     max_sharpe_idx = np.argmax(results[2])
#     optimal_weights = weights_record[max_sharpe_idx]
#     optimal_return = results[0, max_sharpe_idx]
#     optimal_volatility = results[1, max_sharpe_idx]
#     optimal_sharpe = results[2, max_sharpe_idx]

#     # 4. Plot efficient frontier
#     fig, ax = plt.subplots()
#     sc = ax.scatter(results[1, :], results[0, :],
#                     c=results[2, :], cmap='viridis', alpha=0.7)
#     ax.scatter(optimal_volatility, optimal_return, c='red',
#                marker='*', s=150, label='Max Sharpe')
#     ax.set_xlabel('Annualized Volatility')
#     ax.set_ylabel('Annualized Return')
#     ax.set_title('Efficient Frontier')
#     ax.legend()
#     fig.colorbar(sc, label='Sharpe Ratio')

#     # Save plot to HTML string
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     plt.close(fig)
#     buffer.seek(0)
#     image_png = buffer.getvalue()
#     buffer.close()
#     plot_url = base64.b64encode(image_png).decode('utf-8')
#     img_html = f'<img src="data:image/png;base64,{plot_url}" alt="Efficient Frontier" class="mt-4"/>'

#     # 5. Format weights table
#     weights_df = pd.DataFrame({
#         'Ticker': tickers,
#         'Weight': [f"{w:.2%}" for w in optimal_weights]
#     })
#     weights_html = weights_df.to_html(
#         index=False, classes="table-auto w-full text-left whitespace-no-wrap")

#     # 6. Summary metrics
#     summary_html = f"""
#         <p><strong>Expected Return:</strong> {optimal_return:.2%}</p>
#         <p><strong>Volatility:</strong> {optimal_volatility:.2%}</p>
#         <p><strong>Sharpe Ratio:</strong> {optimal_sharpe:.2f}</p>
#     """

#     return weights_html + summary_html + img_html


# def mean_variance_optimization(df_returns):
#     """
#     Performs Mean-Variance Optimization and returns HTML results and efficient frontier image.
#     """
#     mu = df_returns.mean() * 252  # annualized return
#     cov = df_returns.cov() * 252  # annualized covariance
#     num_assets = len(mu)

#     results = []
#     weights_record = []

#     for _ in range(5000):
#         weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
#         ret = np.dot(weights, mu)
#         vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
#         sharpe = ret / vol
#         results.append([ret, vol, sharpe])
#         weights_record.append(weights)

#     results = np.array(results)
#     weights_record = np.array(weights_record)

#     max_sharpe_idx = np.argmax(results[:, 2])
#     optimal_weights = weights_record[max_sharpe_idx]
#     optimal_return = results[max_sharpe_idx, 0]
#     optimal_vol = results[max_sharpe_idx, 1]
#     optimal_sharpe = results[max_sharpe_idx, 2]

#     # Create plot
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sc = ax.scatter(results[:, 1], results[:, 0],
#                     c=results[:, 2], cmap='viridis', alpha=0.5)
#     ax.scatter(optimal_vol, optimal_return, marker='*',
#                color='r', s=200, label='Max Sharpe')
#     ax.set_xlabel('Volatility (Risk)')
#     ax.set_ylabel('Expected Return')
#     ax.set_title('Efficient Frontier')
#     ax.legend()
#     plt.colorbar(sc, label='Sharpe Ratio')

#     # Save plot to base64
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     plt.close(fig)
#     buf.seek(0)
#     image_base64 = base64.b64encode(buf.read()).decode('utf-8')

#     # Construct results HTML
#     metrics_html = f"""
#     <h3 class="text-lg font-bold text-green-700 mb-2">Mean-Variance Optimization Results</h3>
#     <ul class="list-disc ml-6 text-gray-700">
#         <li><strong>Max Sharpe Ratio:</strong> {optimal_sharpe:.4f}</li>
#         <li><strong>Expected Annual Return:</strong> {optimal_return:.2%}</li>
#         <li><strong>Annual Volatility:</strong> {optimal_vol:.2%}</li>
#     </ul>
#     <h4 class="mt-4 font-semibold">Optimal Portfolio Weights:</h4>
#     <table class="table-auto w-full mt-2 text-left text-sm text-gray-800 border border-gray-300">
#         <thead><tr><th class="border px-2">Asset</th><th class="border px-2">Weight</th></tr></thead>
#         <tbody>
#     """
#     for asset, weight in zip(df_returns.columns, optimal_weights):
#         metrics_html += f"<tr><td class='border px-2'>{asset}</td><td class='border px-2'>{weight:.2%}</td></tr>"
#     metrics_html += "</tbody></table>"

#     img_html = f'<img src="data:image/png;base64,{image_base64}" class="mt-6 rounded shadow-md"/>'

#     return metrics_html + img_html
