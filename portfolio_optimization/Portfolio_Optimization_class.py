#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For ECON369 Spring 2026 Assignment 1
@author: raiyat
"""

import numpy as np
import pandas as pd
import statistics
from statsmodels import robust
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
import quantstats as qs
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime

## class for all asset data
class asset_return_data_class:
    def __init__(self, file_dir, T, is_portfolio,
                 ticker_list, ticker_rf, ticker_index_list,
                 start_date=datetime(2010,1,1).date(), 
                 end_date=datetime(2030,12,31).date()):
        self.file_dir = file_dir
        self.T = T
        self.ticker_list = ticker_list
        self.ticker_rf = ticker_rf
        self.ticker_index_list = ticker_index_list
        self.start_date = start_date
        self.end_date = end_date
        
        self.alpha_list = [0.95, 0.99] # VaR confidence level
        self.var_target_list = [0.05, 0.1] # VaR target
        
        if not is_portfolio: # asset level
            self.asset_return_df = self.asset_return(index=False)
            self.asset_return_index_df = self.asset_return(index=True)   
            self.asset_return_df_rf = self.asset_return_rf()
            self.asset_return_df, self.asset_return_df_rf, self.asset_return_combined = self.asset_return_scaled()
        
            self.asset_return_df.to_csv(file_dir+'asset_return.csv', index=True, index_label='date')
            self.asset_return_index_df.to_csv(file_dir+'asset_return_index.csv', index=True, index_label='date')  
            self.asset_return_df_rf.to_csv(file_dir+'asset_return_rf.csv', index=True, index_label='date')
            self.asset_return_combined.to_csv(file_dir+'asset_return_combined.csv', index=True, index_label='date')
           
            self.asset_des_stats = self.asset_return_combined.apply(self.asset_return_des_stats, axis=0)
            self.asset_des_stats.to_csv(file_dir+'asset_des_statistics.csv', index=True, index_label='date')
            
            self.asset_return_df_cov_list = self.asset_return_cov_list()
            self.asset_return_df_cov_list.to_csv(file_dir+'asset_return_cov.csv', index=True, index_label='date')
        else: # portfolio level 
            self.asset_return_df = ticker_list
            self.asset_return_index_df = ticker_index_list
            self.asset_return_df_rf = ticker_rf
            self.asset_return_combined = pd.concat([self.asset_return_df, self.asset_return_df_rf], axis=1)
                       
            self.asset_des_stats = self.asset_return_combined.apply(self.asset_return_des_stats, axis=0)
               
    ## functions to get return series, stats, and covariance series
    def asset_return(self, index): # function to get the asset return dataframe
        if index: # ticker is the index
            ticker_list = self.ticker_index_list
        else: # ticker is the list of stocks    
            ticker_list = self.ticker_list
        
        cut_off = self.start_date
        
        ticker_return = []
        for ticker in ticker_list.keys():
            ticker = yf.Ticker(ticker)
            #ticker_df = ticker.history(period='100y') # up to 100 years of historical data
            ticker_df = ticker.history(period='max') # maximum amount of allowed historical data
            ticker_df['Return'] = ticker_df['Close'].pct_change().dropna(how='any')        
            ticker_df.index = pd.to_datetime(ticker_df.index).tz_localize(None).date       
            ticker_return.append(ticker_df['Return'])
        
        ticker_return = pd.concat(ticker_return, axis=1).ffill().dropna(how='any')
        ticker_return = ticker_return.loc[ticker_return.index>cut_off].sort_index()
        ticker_return.columns = ticker_list.values()
        
        return ticker_return
    
    def asset_return_rf(self):
        T = self.T
        ticker_rf = self.ticker_rf
        start_date = self.start_date
        end_date = self.end_date
    
        ticker_return = web.DataReader(ticker_rf, 'fred', start_date, end_date)/100/T
        ticker_return.index = pd.to_datetime(ticker_return.index).tz_localize(None).date 
        
        return ticker_return
    
    # line up stocks and risk-free dates    
    def asset_return_scaled(self):
        ticker_rf = self.ticker_rf
        ticker_df = self.asset_return_df
        ticker_df_rf = self.asset_return_df_rf
        
        # find the first common date
        first_overlap = ticker_df.index.intersection(ticker_df_rf.index).min()

        # trim both dfs from that date forward, then combine, treat unmated dates
        ticker_df_combined = ticker_df.loc[ticker_df.index>=first_overlap].join(ticker_df_rf.loc[ticker_df_rf.index>=first_overlap], how="outer")
        # some rows have not data for stocks and risk free
        ticker_df_combined = ticker_df_combined.dropna(how='all')       
        # some rows have data from stocks, but not for risk free, keep and forward fill na
        ticker_df_combined[ticker_rf] = ticker_df_combined[ticker_rf].ffill()
        # some rows have data for risk free, but not for stocks, drop
        ticker_df_combined = ticker_df_combined.dropna(how='any') 
        
        ticker_df = ticker_df.loc[ticker_df_combined.index]
        ticker_df_rf = ticker_df_combined[ticker_rf].to_frame()

        return ticker_df, ticker_df_rf, ticker_df_combined
        
    # function to get the asset return descriptive statistics
    def asset_return_des_stats(self, ticker_df): 
        alpha_list = self.alpha_list
        var_target_list = self.var_target_list
        #print(ticker_df.name)
        
        ticker_skewness = ticker_df.skew()
        ticker_kurtosis = stats.kurtosis(ticker_df, fisher=True)
        ticker_describe = ticker_df.describe()
                
        # A) Given confidence, get VaR & CVaR
        ticker_var_list = []
        ticker_cvar_list = []
        for alpha in alpha_list:
            var, cvar, a, t, mu, sigma = self.var_cvar_parametric(ticker_df, alpha=alpha)
            # print(ticker_df.name)
            # print(f"{int(a*100)}% conf (tail {t*100:.2f}%): VaR={var:.2%}, CVaR={cvar:.2%}")
            self.plot_var_cvar(ticker_df.name, mu, sigma, var, cvar, a, a_to_var=True)
            ticker_var_list.append(var)
            ticker_cvar_list.append(cvar)
            
        # B) Given a desired VaR magnitude (e.g., 5% loss), infer alpha and return VaR & CVaR
        ticker_var_t_list = []
        ticker_cvar_t_list = []
        ticker_a_t_list = []
        for var_target in var_target_list:
            var_t, cvar_t, a_t, t_t, mu_t, sigma_t = self.var_cvar_parametric(ticker_df, var_loss=var_target)
            # print(f"Target VaR={var_target:.2%} → alpha={a_t:.2%}, tail={t_t*100:.2f}%, CVaR={cvar_t:.2%}")
            self.plot_var_cvar(ticker_df.name, mu_t, sigma_t, var_t, cvar_t, a_t, a_to_var=False)
            ticker_var_t_list.append(var_t)
            ticker_cvar_t_list.append(cvar_t)
            ticker_a_t_list.append(a_t)
            
        ticker_misc = pd.DataFrame([ticker_skewness, ticker_kurtosis, 
                                    ticker_var_list[0], ticker_cvar_list[0], 
                                    ticker_var_list[1], ticker_cvar_list[1],
                                    ticker_var_t_list[0], ticker_cvar_t_list[0], ticker_a_t_list[0],
                                    ticker_var_t_list[1], ticker_cvar_t_list[1], ticker_a_t_list[1]], 
                                    index=['skewness', 'kurtosis',
                                           'var_95', 'cvar_95', 
                                           'var_99', 'cvar_99',                                            
                                           'var_t_005', 'cvar_t_005', 'alpha_t_005',
                                           'var_t_01', 'cvar_t_01', 'alpha_t_01'],
                                    columns=[ticker_df.name])
            
        ticker_des_stats = pd.concat([ticker_describe.to_frame(), ticker_misc], axis=0)[ticker_df.name]

        return ticker_des_stats
    
    def alpha_for_var(self, returns, var_loss):
        """Given a desired VaR magnitude (var_loss > 0), compute the confidence level alpha under Normal assumption."""
        s = pd.Series(returns).dropna()
        mu, sigma = s.mean(), s.std(ddof=1)
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive.")
        L = float(var_loss)
        alpha = norm.cdf((L + mu) / sigma)
        tail_prob = 1 - alpha
        return alpha, tail_prob, mu, sigma
    
    # function to get var and cvar
    ## A) given 95% and 99% confidence level calculat var and cvar
    ## B) given 5% and 1% var, calculate cvar and confidence level
    def var_cvar_parametric(self, returns, alpha=None, var_loss=None):
        """
        Parametric (Normal) VaR & CVaR.
        - Use either alpha (confidence, e.g. 0.95) OR var_loss (target VaR magnitude, e.g. 0.05).
        Returns: (VaR, CVaR, alpha, tail_prob, mu, sigma)
        """
        #print("am I here?")
        if (alpha is None) == (var_loss is None):
            raise ValueError("Provide exactly one of alpha or var_loss.")
    
        s = pd.Series(returns).dropna()
        mu, sigma = s.mean(), s.std(ddof=1)
        if sigma <= 0:
            raise ValueError("Standard deviation must be positive.")
    
        if var_loss is not None:
            alpha, tail_prob, _, _ = self.alpha_for_var(s, var_loss)
            var_ = float(var_loss)
        else:
            tail_prob = 1 - float(alpha)
            z = norm.ppf(1 - alpha)
            var_ = -(mu + z * sigma)
    
        z = norm.ppf(1 - alpha)
        cvar_ = -(mu - sigma * norm.pdf(z) / (1 - alpha))
    
        return var_, cvar_, alpha, tail_prob, mu, sigma
    
    # -----------------------------
    # Seaborn Plotting Function
    # -----------------------------
    def plot_var_cvar(self, ticker_name, mu, sigma, var, cvar, alpha, a_to_var=True):
        file_dir = self.file_dir
        """
        Plot the normal distribution with VaR and CVaR using Seaborn.
        """
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        y = norm.pdf(x, mu, sigma)
    
        plt.figure(figsize=(5,3))
        sns.lineplot(x=x, y=y, color="blue", label="Normal distribution")
    
        # Shade tail region beyond VaR
        x_fill = np.linspace(mu - 4*sigma, -var, 500)
        y_fill = norm.pdf(x_fill, mu, sigma)
        plt.fill_between(x_fill, y_fill, color="red", alpha=0.3,
                         label=f"Worst {int((1-alpha)*100)}% tail")
    
        # VaR line
        plt.axvline(-var, color="red", linestyle="--", linewidth=2,
                    label=f"VaR {int(alpha*100)}% = {var:.2%}")
    
        # CVaR line
        plt.axvline(-cvar, color="darkred", linestyle=":", linewidth=2,
                    label=f"CVaR {int(alpha*100)}% = {cvar:.2%}")
    
        if a_to_var:
            plt.title(f"{ticker_name} VaR and CVaR at {int(alpha*100)}% Confidence")
            output_path = file_dir + f'{ticker_name}_VaR_CVaR_at_{int(alpha*100)}%_Confidence'           
        else:
            plt.title(f"{ticker_name} CVaR and {int(alpha*100)}% Confidence at VaR")
            output_path = file_dir + f'{ticker_name}_CVaR_and_{int(alpha*100)}%_Confidence_at_VaR'
        
        plt.xlabel("Return")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
         
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show() 
        
        return True
    
    def asset_return_cov_list(self):
        T = self.T
        df = self.asset_return_df
        df_cov_list = df.rolling(window=T).cov().dropna(how='all', axis=0)
    
        # Unique dates (ascending)
        dates = df_cov_list.index.get_level_values(0).unique()#.sort_values()
    
        cols = list(df_cov_list.columns)
        cov_list = []
        for d in dates:
            # d = dates[0]
            blk = df_cov_list.loc[(d, slice(None)), :]  # all rows for this date
            sigma = blk.droplevel(0)                    # rows -> assets
            # align to consistent asset order (keeps matrices square & aligned)
            sigma = sigma.loc[cols, cols]
            cov_list.append(sigma)
    
        cov_list = pd.Series(cov_list, index=dates, name="covariance").to_frame()
    
        return cov_list

class efficient_frontier_class:
    def __init__(self, file_dir, T, cap, rf_weight, lambdas,
                 asset_return_df, asset_return_df_cov_list, asset_return_df_rf, 
                 cov_is_daily):
        self.file_dir = file_dir
        self.T = T
        self.cap = cap
        self.rf_weight = rf_weight
        self.lambdas = lambdas
        self.cov_is_daily = True
        self.asset_return_df = asset_return_df
        self.asset_return_df_cov_list = asset_return_df_cov_list
        self.asset_return_df_rf = asset_return_df_rf
        
        self.results_by_date, self.weights_df = self.per_date_frontiers_capped()
        self.results_by_date = self.attach_lambda_sweep_to_results()
        
        self.portfolio_df_dict = self.extract_portfolio_dataframes()
    
    ## functions for efficient frontier
    # ---- primitives (long-only, capped) ----
    def _gmv_capped(self, S, cap=0.5):
        n = S.shape[0]; b = [(0.0, cap)]*n; c = ({"type":"eq","fun":lambda w: np.sum(w)-1.0},)
        f = lambda w: float(w.reshape(-1,1).T @ S @ w.reshape(-1,1))
        r = minimize(f, np.ones(n)/n, method="SLSQP", bounds=b, constraints=c)
        if not r.success: raise RuntimeError("GMV failed: "+r.message)
        
        return r.x
    
    def _minvar_for_target(self, mu, S, R, cap=0.5):
        n = len(mu); b = [(0.0, cap)]*n
        c = ({"type":"eq","fun":lambda w: np.sum(w)-1.0},
             {"type":"ineq","fun":lambda w, R=R: mu@w - R})
        f = lambda w: float(w.reshape(-1,1).T @ S @ w.reshape(-1,1))
        r = minimize(f, np.ones(n)/n, method="SLSQP", bounds=b, constraints=c)
        
        return r.x if r.success else None
    
    def _perf(self, w, mu, S, rf=0.0):
        r = float(w@mu); v = float(np.sqrt(w@S@w)); s = (r-rf)/v if v>0 else np.nan
        
        return r,v,s
    
    # ---- per-date frontier ----
    def per_date_frontiers_capped(self):
        T = self.T
        cap = self.cap
        cov_is_daily = self.cov_is_daily
        returns_df = self.asset_return_df
        cov_by_date = self.asset_return_df_cov_list
        rf_df = self.asset_return_df_rf
        
        # returns_df = asset_return_df
        # cov_by_date = asset_return_df_cov_list
        # rf_df = asset_return_df_rf
        lookback = T            
        freq = T #if cov_by_date else 1.0                                                                  
        n_points = 101
    
        # returns: (results_by_date, weights_df)
        df = returns_df.copy()
        dates = (df.index).intersection(cov_by_date.index)
        # dates = dates[-T:] # short time series for quick calculation'
        assets = list(df.columns)
    
        recs, results_by_date = [], {}
        # each date in the dates
        for dt in dates:
            # dt = dates[0]
            print(f"on date {dt}")
            rf = rf_df.loc[dt].values[0]
            win = df.loc[:dt].tail(lookback)
            if win.empty: continue
            mu = win.mean().values * freq
            Sdf = cov_by_date.loc[dt]
            S = (Sdf.values * (freq if cov_is_daily else 1.0))[0].values
    
            # GMV
            w_gmv = self._gmv_capped(S, cap)
            rmin, rmax = float(np.min(mu)), float(np.max(mu)*2)
            targets = np.linspace(rmin, rmax, n_points)
    
            rows, wlist = [], []
            for R in targets:
                w = self._minvar_for_target(mu, S, R, cap)
                if w is None: continue
                r,v,s = self._perf(w, mu, S, rf)
                rows.append({"target_return": float(R), "return": r, "vol": v, "sharpe": s})
                wlist.append(w)
    
            frontier = pd.DataFrame(rows)
            frontier["weights"] = wlist
            w_tan = frontier.loc[frontier["sharpe"].idxmax(), "weights"] if not frontier.empty else w_gmv
    
            # scalar μ, σ, and Sharpe at tangent point + GMV (optional) ---
            r_g, v_g, s_g = self._perf(w_gmv, mu, S, rf)   # optional, kept for completeness
            r_t, v_t, s_t = self._perf(w_tan, mu, S, rf)   # tan_ret, tan_vol, tan_sharpe
            cml_sharpe = s_t                          # CML slope equals Sharpe at tangent point
    
            # store weights (GMV + frontier)
            recs.append({"date":dt,"kind":"GMV","point_id":0, **dict(zip(assets, w_gmv))})
            for pid, w in enumerate(wlist):
                recs.append({"date":dt,"kind":"FRONTIER","point_id":pid, **dict(zip(assets, w))})
    
            # store results
            results_by_date[dt] = {
                "assets": assets,
                "mu": mu,                 # vector
                "sigma": S,               # matrix
                "frontier": frontier,
                "w_gmv": w_gmv,
                "w_tan": w_tan,
                "tan_return": r_t,           # <-- scalar μ at tangent point
                "tan_volatility": v_t,           # <-- scalar σ at tangent point
                "tan_sharpe": s_t,        # <-- Sharpe at tangent point
                "cml_sharpe": cml_sharpe  # <-- same as tan_sharpe
            }
    
        weights_df = pd.DataFrame.from_records(recs).set_index(["date","kind","point_id"]).sort_index()
        
        return results_by_date, weights_df
    
    ## now use full uw - λ/2 w*SIGMA*w
    ## with sweeping lambda from 0.5 to 5
    # =========================
    # Quadratic-Utility (λ-sweep)
    # =========================
    def solve_quadratic_utility(self, mu, S, rf=0.0, cap=0.5, lambdas=None):
        """
        Solve, for each lambda in the grid:
            max_w   w^T mu - (lambda/2) * w^T S w
            s.t.    sum w = 1,   0 <= w_i <= cap
    
        Parameters
        ----------
        mu : np.ndarray
            Expected returns vector (units consistent with S and rf).
        S : np.ndarray
            Covariance matrix (same units as mu).
        rf : float
            Risk-free rate (same units as mu).
        cap : float
            Per-asset cap (e.g., 0.5 means each weight <= 50%).
        lambdas : iterable or None
            Risk-aversion parameters. If None, uses 0.5..5.0 step 0.5.
    
        Returns
        -------
        pd.DataFrame with columns:
            ['lambda','return','vol','sharpe','weights']
        """
        if lambdas is None:
            lambdas = np.arange(0.5, 5.0 + 1e-12, 0.5)
    
        n = len(mu)
        bounds = [(0.0, cap)] * n
        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    
        rows = []
        for lam in lambdas:
            # We minimize the negative of utility to use SLSQP
            def obj(w):
                return -(w @ mu - 0.5 * lam * (w @ S @ w))
    
            res = minimize(obj, np.ones(n)/n, method="SLSQP", bounds=bounds, constraints=cons)
            if not res.success:
                # skip infeasible or failed point
                continue
    
            w = res.x
            r = float(w @ mu)
            v = float(np.sqrt(w @ S @ w))
            s = (r - rf) / v if v > 0 else np.nan
            rows.append({"lambda": float(lam), "return": r, "vol": v, "sharpe": s, "weights": w})
    
        return pd.DataFrame(rows)
    
    def attach_lambda_sweep_to_results(self):
        results_by_date = self.results_by_date
        rf_df = self.asset_return_df_rf
        cap = self.cap 
        lambdas = self.lambdas
        
        key = "lambda_sweep"
        """
        For each date in results_by_date, compute the λ-sweep (quadratic utility)
        and store the DataFrame under results_by_date[dt][key].
    
        Notes:
        - This assumes your existing dict uses:
            info["mu"]    -> the expected returns vector
            info["sigma"] -> the covariance matrix (n x n)
        - Units must match (annualized vs daily).
        """
        for dt, info in results_by_date.items():
            mu = info["mu"]
            S  = info["sigma"]   # keep your existing key name
            rf = rf_df.loc[dt].values[0]
            df_lambda = self.solve_quadratic_utility(mu, S, rf=rf, cap=cap, lambdas=lambdas)
            results_by_date[dt][key] = df_lambda
            
        return results_by_date
    
    def extract_portfolio_dataframes(self):
        """
        Build tidy DataFrames for GMV, Tangency, and each λ in self.lambdas,
        adding a risk-free asset column (e.g., 'SOFR') and scaling risky weights
        to sum to (1 - rf_weight). Metrics (return/vol/sharpe) are recomputed for
        the mixed portfolio:
            r_mix = rf_weight*rf + (1-rf_weight)*r_risky
            v_mix = (1-rf_weight)*v_risky
            s_mix = (r_mix - rf) / v_mix = same Sharpe as risky portfolio
        Returns
        -------
        dict of DataFrames with keys:
          - "gmv"
          - "tan"
          - "lambda_<val>"
        """
        results_by_date = self.results_by_date
        asset_return_df = self.asset_return_df
        rf_df = self.asset_return_df_rf
        rf_weight = self.rf_weight
    
        assets = list(asset_return_df.columns)
        # risk-free column name (use your SOFR column name if present)
        rf_col = rf_df.columns[0] if hasattr(rf_df, "columns") and len(rf_df.columns) else "rf"
    
        # robust lambda iterator
        if self.lambdas is None:
            lambdas_iter = []
        else:
            lambdas_iter = list(np.asarray(self.lambdas, dtype=float).ravel())
    
        out_g, out_t = [], []
        out_lambda = {float(l): [] for l in lambdas_iter}
    
        for dt in sorted(results_by_date.keys()):
            info = results_by_date[dt]
            mu, S = info["mu"], info["sigma"]
            w_g, w_t = info["w_gmv"], info["w_tan"]
    
            # scalar rf for date dt
            rf = float(np.asarray(rf_df.loc[dt]).squeeze())
    
            # --- helper: mix risky weights with rf and recompute metrics ---
            def mixed_row(w_risky, label_assets=assets):
                # risky-only stats
                r_r, v_r, s_r = self._perf(w_risky, mu, S, rf)
                # mixed stats
                r_m = rf_weight*rf + (1.0 - rf_weight)*r_r
                v_m = (1.0 - rf_weight)*v_r
                s_m = (r_m - rf) / v_m if v_m > 0 else np.nan
                # weights: scale risky, add rf column
                w_scaled = (1.0 - rf_weight) * w_risky
                row = {"date": dt, "return": r_m, "vol": v_m, "sharpe": s_m}
                row.update({a: float(wi) for a, wi in zip(label_assets, w_scaled)})
                row[rf_col] = float(rf_weight)
                return row
    
            # GMV mixed row
            out_g.append(mixed_row(w_g))
    
            # Tangency mixed row (use stored scalars if you want, but recompute is fine)
            out_t.append(mixed_row(w_t))
    
            # λ-sweep rows (if present)
            df_lam = info.get("lambda_sweep")
            if df_lam is not None and len(df_lam) and lambdas_iter:
                lam_col = df_lam["lambda"].to_numpy(dtype=float)
                for lam in lambdas_iter:
                    m = np.isclose(lam_col, float(lam), rtol=0.0, atol=1e-10)
                    if not m.any():
                        continue
                    r = df_lam.loc[m].iloc[0]
                    w = r["weights"]  # risky-only weights (sum=1)
                    row_l = mixed_row(w)
                    out_lambda[float(lam)].append(row_l)
    
        # assemble dict of DataFrames
        df_dict = {
            "gmv": pd.DataFrame(out_g).set_index("date").sort_index(),
            "tan": pd.DataFrame(out_t).set_index("date").sort_index(),
        }
        
        for lam, rows in out_lambda.items():
            key = f"lambda_{lam:g}"
            if rows:
                df_dict[key] = pd.DataFrame(rows).set_index("date").sort_index()
            else:
                # ensure columns present even if empty
                df_dict[key] = pd.DataFrame(columns=[*assets, rf_col, "return", "vol", "sharpe"])
    
        return df_dict  
    
class portfolio_optimization_class:
    def __init__(self, file_dir, portfolio_df_dict,
                 asset_return_combined, asset_return_df_rf,
                 asset_return_df_cov_list):
        
        """
        portfolio_df_dict : dict[str -> DataFrame] weights by date (risky + rf)
        returns_all_df    : DataFrame daily returns (risky + rf)
        cov_series_df     : 1-col DataFrame; each cell is (n x n) Σ for risky assets
        rf_col            : risk-free column name in returns/weights
        """

        self.file_dir = file_dir
        self.portfolios_dict = portfolio_df_dict
        self.r_all = asset_return_combined
        self.cov = asset_return_df_cov_list
        self.rf_df = asset_return_df_rf
        self.rf_col = asset_return_df_rf.columns[0]
        self._cov_name = asset_return_df_cov_list.columns[0]
        
        self.shift = 1
        self.portfolio_performance_dict = self.compute_all() 

    def compute_all(self):
        portfolios_dict = self.portfolios_dict
        # portfolios_dict = portfolio_optimization.portfolios_dict
        order = [k for k in ("gmv","tan") if k in portfolios_dict]
        order += [k for k in portfolios_dict if k.startswith("lambda_") and k not in order]
        
        portfolio_performance_dict = {k: self._one(k) for k in order}
        
        return portfolio_performance_dict

    def _one(self, key):
        r_all, cs, rf, cov_name = self.r_all, self.cov, self.rf_col, self._cov_name
        portfolios_dict = self.portfolios_dict
        shift = self.shift
        
        # r_all = portfolio_optimization.r_all
        # cs = portfolio_optimization.cov
        # rf = portfolio_optimization.rf_col
        # cov_name = portfolio_optimization._cov_name
        # portfolios_dict = portfolio_optimization.portfolios_dict  
        
        wdf = portfolios_dict[key].copy()
        wdf.drop(columns=[c for c in ("return","vol","volatility","sharpe") if c in wdf.columns], inplace=True, errors="ignore")

        idx = r_all.index.union(wdf.index).union(cs.index)
        r   = r_all.reindex(idx)
        w   = wdf.reindex(idx).ffill()
        cs_ = cs.reindex(idx)

        risky_cols = [c for c in w.columns if c != rf and c in r.columns]

        # port_ret = (w[risky_cols] * r[risky_cols].fillna(0.0)).sum(axis=1)
        port_ret = (w[risky_cols].shift(shift).fillna(0.0) * r[risky_cols].fillna(0.0)).sum(axis=1)
        
        if rf in w.columns and rf in r.columns:
            port_ret = port_ret + w[rf].shift(shift) * r[rf].fillna(0.0)

        w_shift = w.shift(shift)
        vols = []
        for dt in cs_.index:
            # dt = cs_.index[0]
            Sigma = cs_.at[dt, cov_name]
            if Sigma is None or (isinstance(Sigma, float) and np.isnan(Sigma)) or not risky_cols:
                vols.append(np.nan); continue
            wr = w_shift.loc[dt, risky_cols].to_numpy(float)
            if isinstance(Sigma, pd.DataFrame):
                try: Sigma = Sigma.reindex(index=risky_cols, columns=risky_cols).to_numpy()
                except: Sigma = Sigma.to_numpy()
            else:
                Sigma = np.asarray(Sigma)
            vols.append(np.sqrt(wr @ Sigma @ wr) if Sigma.shape == (len(wr), len(wr)) else np.nan)

        out = w.copy()
        out["return"]     = port_ret
        out["volatility"] = pd.Series(vols, index=cs_.index)

        keep = r.index.intersection(cs_.index)
        out = out.loc[keep]
        base = [c for c in out.columns if c not in ("return","volatility")]
        
        return out[base + ["return","volatility"]]
 
class portfolio_performance_class:
    def __init__(self, file_dir, T,
                 portfolio_performance_dict, return_index_df, rf_df):
        self.start = 100.0 # initial portfolio wealth
        self.file_dir = file_dir
        self.T = T
        self.portfolio_performance_dict = portfolio_performance_dict
        self.return_index_df = return_index_df
        self.rf_df = rf_df
        
        # calculating performance metrics
        self.returns_df, self.rf_df, self.return_index_df = self.combine_returns()
        self.portfolio_weight_dict = self.extract_weights()
        self.prices_df = self.compute_prices()
        
        self.sharpe_ratio_df = self.compute_sharpe()
        self.sortino_ratio_df = self.compute_sortino()
        self.drawdowns_df = self.compute_drawdowns()
        self.max_drawdown_df = self.compute_max_drawdown()
        
        is_portfolio = True
        self.portfolio_stats = asset_return_data_class(
            file_dir, T, is_portfolio,
            self.returns_df,self.rf_df, self.return_index_df
            )
        
        self.performance_table = self.metrics_table(annualize=True)
        self.performance_table = pd.concat([self.portfolio_stats.asset_des_stats, 
                                            self.performance_table], axis=0)
        self.performance_table.to_csv(file_dir+'performance_table.csv', index=True)
        
        self._prices_plot = self.plot_prices()
        self._returns_plot = self.plot_returns()
        self._drawdown_plot = self.plot_drawdowns()
                
    # line up the data
    def combine_returns(self):
        file_dir = self.file_dir
        portfolio_performance_dict = self.portfolio_performance_dict
        rf_df = self.rf_df
        return_index_df = self.return_index_df
        
        dfs = []
        for key, df in portfolio_performance_dict.items():
            if "return" not in df.columns:
                continue
            dfs.append(df["return"].rename(key))
    
        if not dfs:
            return pd.DataFrame(), rf_df
        
        dfs = pd.concat(dfs, axis=1)
        dfs = pd.concat([dfs, return_index_df], axis=1).sort_index().dropna(how="any")
        rf_dfs = rf_df.loc[dfs.index]
        index_dfs = return_index_df.loc[dfs.index]
        
        dfs.to_csv(file_dir+'portfolio_return.csv', index=True, index_label='date')
        
        return dfs, rf_dfs, index_dfs

    # weigths for each portfolio
    def extract_weights(self):
        file_dir = self.file_dir
        """
        Return a dict of DataFrames:
          { portfolio_name: DataFrame of daily weights (stocks + risk-free) }
        """
        portfolio_performance_dict = self.portfolio_performance_dict
        
        weights_dict = {}
        for key, df in portfolio_performance_dict.items():
            # keep only asset weights (drop return/vol if present)
            cols = [c for c in df.columns if c not in ("return", "volatility", "sharpe")]
            df_weight = df[cols].copy().dropna(how='all')
            weights_dict[key] = df_weight
            df_weight.to_csv(file_dir+key+'_portfolio_weight.csv', index=True, index_label='date')

        return weights_dict

    def compute_prices(self):
        file_dir = self.file_dir
        returns_df = self.returns_df
        start = self.start
        # Start all price series at 100
        prices_df = (1 + returns_df).cumprod() * start
        prices_df.to_csv(file_dir+'portfolio_price.csv', index=True, index_label='date')
        return prices_df

    def compute_sharpe(self):
        returns_df = self.returns_df
        rf_df = self.rf_df
        excess = returns_df.sub(rf_df.squeeze(), axis=0)
        sharpe_ratios = excess.mean() / excess.std(ddof=1)
        return sharpe_ratios

    def compute_sortino(self):
        returns_df = self.returns_df
        rf_df = self.rf_df
        excess = returns_df.sub(rf_df.squeeze(), axis=0)
        downside = excess[excess < 0]
        sortino_ratios = excess.mean() / downside.std(ddof=1)
        return sortino_ratios
    
    def compute_drawdowns(self):
        file_dir = self.file_dir
        prices = self.prices_df
        rolling_max = prices.cummax()
        drawdowns = prices / rolling_max - 1
        drawdowns.to_csv(file_dir+'portfolio_drawdown.csv', index=True, index_label='date')
        return drawdowns

    def compute_max_drawdown(self):
        prices = self.prices_df
        roll_max = prices.cummax()
        drawdowns = (prices - roll_max) / roll_max
        max_drawdowns = drawdowns.min()
        return max_drawdowns
    
    def metrics_table(self, annualize=False):        
        """
        Return a compact metrics table with rows = ['Sharpe','Sortino','MaxDrawdown']
        and columns = portfolio names (same as in self.returns_df).
        Set annualize=True to annualize Sharpe/Sortino using self.T.
        """
        T = self.T
        returns_df = self.returns_df
        sharpe = self.sharpe_ratio_df.copy()
        sortino = self.sortino_ratio_df.copy()
        mdd = self.max_drawdown_df.copy()  # already negative numbers
    
        if annualize:
            return_annualized = returns_df.mean(skipna=True) * T
            vol_annualized = returns_df.std(skipna=True, ddof=1) * np.sqrt(T)
            k = np.sqrt(T)
            sharpe  = sharpe * k
            sortino = sortino * k
    
        tbl = pd.DataFrame({
            "Annualized Return": return_annualized,
            "Annualized Volatility": vol_annualized,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max Drawdown": mdd
        }).T

        # clean up any inf/-inf from zero stdev cases
        return tbl.replace([np.inf, -np.inf], np.nan)

    def plot_prices(self):
        file_dir = self.file_dir
        prices_df = self.prices_df
        
        title = "Portfolio and Index Prices"
        prices_df.plot(figsize=(10, 6), title=title)
        plt.ylabel("Price")
        plt.grid(True) 
        plt.savefig(file_dir+title, dpi=300, bbox_inches="tight")
        plt.show() 

        return True
    
    def plot_returns(self):
        file_dir = self.file_dir
        returns_df = self.returns_df
        
        title = "Portfolio and Index Returns"
        returns_df.plot(figsize=(10, 6), title=title)
        plt.ylabel("Return")
        plt.grid(True) 
        plt.savefig(file_dir+title, dpi=300, bbox_inches="tight")
        plt.show() 

        return True

    def plot_drawdowns(self):
        file_dir = self.file_dir
        drawdowns_df = self.drawdowns_df
        
        title = "Portfolio and Index Drawdowns"
        drawdowns_df.plot(figsize=(10, 6), title=title)
        plt.ylabel("Drawdown")
        plt.grid(True)
        plt.savefig(file_dir+title, dpi=300, bbox_inches="tight")        
        plt.show() 
        
        return True
#%%        
               