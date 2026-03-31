#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: raiyat
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV
from pandas import Timedelta
from scipy.optimize import minimize

class asset_fama_macbeth_data_class:
    def __init__(self, file_dir, T, T_momentum, 
                 index_name, ticker_list, indicator_list,
                 return_df, return_rf, return_index,
                 ff_carhart_factors, ticker_indicator_dict):
        self.file_dir = file_dir #= work_dir
        self.T = T
        self.T_momentum = T_momentum
        self.index_name = index_name
        self.ticker_list = ticker_list
        self.indicator_list = indicator_list
        
        self.return_df = return_df
        self.return_rf = return_rf
        self.return_index = return_index
        
        self.reduced_ticker_list = {k: v for k, v in ticker_list.items() if v in self.return_df.columns}
        
        self.ff_carhart_factors = ff_carhart_factors                
        self.ticker_indicator_dict = ticker_indicator_dict
        
        self.hac_lags = 4
        self.lag = 1
        
        self.priority = [# preferred economic representatives
            "RSI_T","MACD_hist","STOCHF_K","NATR_T","BB_width","AROON_UP",
            "HT_TRENDMODE","KAMA_T","TYPPRICE","OBV","MFI_T","LINREG_SLOPE_2T",
            "BETA_2T_MKT","CORR_2T_MKT","LOG_DOLLAR_VOL","AMIHUD_2T",
            "LOG_MCAP","TURNOVER","RVOL_2T"]        
        self.corr_thresh = 0.70
        
        self.ticker_reduced_indicator_dict = self.reduce_indicator()

        # fama-macbeth stage 1 - time series
        # asset pricing, no shift of fama-french and carhart factors
        self.betas_pricing, self.tvals_pricing, self.ts_date_pricing = self.fama_macbeth_ts('pricing')
        # predictive, shift fama-french and carhart factors by 1 day
        self.betas_predictive, self.tvals_predictive, self.ts_date_predictive = self.fama_macbeth_ts('predictive')

        # fama-macbeth stage 2 - cross sectional
        # asset pricing
        self.lambda_pricing_dict, self.lambda_pricing_summary, self.lambda_t_pricing = self.fama_macbeth_cs(self.betas_pricing, self.ts_date_pricing, 'pricing')
        # predictive
        self.lambda_predictive_dict, self.lambda_predictive_summary, self.lambda_t_predictive = self.fama_macbeth_cs(self.betas_predictive, self.ts_date_predictive, 'predictive')
        
    def reduce_indicator(self):
        ticker_indicator_dict = self.ticker_indicator_dict
        
        # --- list of indicators to keep (24) ---
        keep_cols = [
            # momentum / trend
            "RSI_T", "MACD_hist", "PPO", "KAMA_T",       
            # stochastics
            "STOCH_K", "STOCHF_K",        
            # volatility
            "NATR_T", "STDDEV_2T", "BB_width", "BB_pctB",       
            # aroon
            "AROON_UP",       
            # trend mode
            "HT_TRENDMODE",        
            # price-type
            "TYPPRICE",        
            # volume-based
            "OBV", "MFI_T",       
            # regression features
            "LINREG_SLOPE_2T",        
            # market relations
            "BETA_2T_MKT", "CORR_2T_MKT",        
            # liquidity / size
            "LOG_DOLLAR_VOL", "AMIHUD_2T", "LOG_MCAP", "TURNOVER",       
            # optional extras
            "RVOL_2T", "FLOAT_TURNOVER"
        ]
        
        # --- simplify the dictionary ---
        ticker_reduced_indicator_dict = {
            t: df.loc[:, df.columns.intersection(keep_cols)] for t, df in ticker_indicator_dict.items()
        }
        
        return ticker_reduced_indicator_dict
        
    def fama_macbeth_ts(self, method):
        file_dir = self.file_dir
        index_name = self.index_name
        # align core
        lag = self.lag
        hac_lags = self.hac_lags
        return_df = self.return_df
        return_rf = self.return_rf
        return_index = self.return_index
        reduced_ticker_list = self.reduced_ticker_list        
        ff_carhart_factors = self.ff_carhart_factors                
        ticker_indicator_dict = self.ticker_indicator_dict
        priority = self.priority
        corr_thresh = self.corr_thresh
        indicator_list = self.indicator_list
        
        alphas = np.logspace(-6, 2, 18)
        coef_tol = 1e-4
        
        idx = return_df.index.intersection(return_rf.index).intersection(return_index.index).intersection(ff_carhart_factors.index)
        return_df = return_df.loc[idx, reduced_ticker_list.values()]
        return_rf = return_rf.loc[idx]
        return_index = return_index.loc[idx]
        
        return_df_ex = return_df.sub(return_rf.values, axis=0)
        return_index_ex = return_index.sub(return_rf.values, axis=0)
        
        factors = pd.concat([return_index_ex, ff_carhart_factors], axis=1)
        # for asset pricing, fama-french and carhart factors don't shift
        # for predictive, shift by 1 day
        if method == 'predictive': factors = factors.shift(lag)
        
        betas, tvals = {}, {}
        ts_date = {}
        for s in reduced_ticker_list.values():
            print(s)
            Z = ticker_indicator_dict[s].loc[idx].shift(lag)#.fillna(0.0)    # T×44 (lagged)
            Z = Z.loc[:, Z.var(ddof=0) > 0]            
                        
            """
            Drop highly correlated columns, keeping one per cluster.
            df : DataFrame (T×K)
            corr_thresh : correlation threshold
            priority : ordered list of preferred columns to keep first
            """
            
            cols = list(Z.columns)
            if priority is not None:
                # reorder so preferred features are considered first
                cols = [c for c in priority if c in Z.columns] + [c for c in cols if c not in priority]
            corr = Z[cols].corr().abs()
            keep = []
            for c in cols:
                if not any(corr.loc[c, keep] > corr_thresh):
                    keep.append(c)
            keep = pd.Index(keep)
            Z = Z[keep]

            # in normal fama-macbeth, regess only TRUE factors
            X = factors.dropna()
            # # if wanted to treat indicators as factors
            # ## not the right way as the indictor values are different 
            # ## asset from asset
            # X = pd.concat([factors, Z], axis=1).dropna()   
            
            Xc = sm.add_constant(X).fillna(0.0) 
            Y = return_df_ex[s].loc[X.index]                
            
            res = sm.OLS(Y.values, Xc.values).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
            
            cols = list(Xc.columns)
            betas[s] = pd.Series(res.params, index=cols, name=s)
            tvals[s] = pd.Series(res.tvalues, index=cols, name=s)

            ts_date[s] = pd.Series(0.0, index=Y.index)                 
                       
        betas = pd.DataFrame(betas).T.fillna(0.0)
        tvals = pd.DataFrame(tvals).T#.fillna(0.0)

        factors_indicators_col = list(factors.columns) + indicator_list        
        # Keep only those that exist, but preserve your desired order
        col_order = [c for c in factors_indicators_col if c in betas.columns]
        betas = betas[col_order]
        tvals = tvals[col_order]
        
        betas.to_csv(file_dir+f'{index_name}_betas_{method}.csv')
        tvals.to_csv(file_dir+f'{index_name}_tvals_{method}.csv')
        
        ts_date = pd.DataFrame(ts_date).dropna(how='any').index
        
        return betas, tvals, ts_date
   
    def fama_macbeth_cs(self, betas, ts_date, method):
        file_dir = self.file_dir
        index_name = self.index_name
        # betas = asset_fama_macbeth_data.betas_pricing
        """
        Stage-2: cross-sectional regressions for each date.
        rex: T×N excess returns
        betas: N×Kf (e.g., Carhart betas)
        ind_by_stock: dict[ticker] -> T×Kc (firm characteristics at t-1)
        tickers: list of tickers
        """
        # align core
        lag = self.lag
        hac_lags = self.hac_lags
        return_df = self.return_df
        return_rf = self.return_rf
        return_index = self.return_index
        reduced_ticker_list = self.reduced_ticker_list        
        ff_carhart_factors = self.ff_carhart_factors                
        # ticker_indicator_dict = self.ticker_indicator_dict
        priority = self.priority
        corr_thresh = self.corr_thresh
        # indicator_list = self.indicator_list
        
        alphas = np.logspace(-6, 2, 18)
        coef_tol = 1e-4
        
        # betas = asset_fama_macbeth_data.betas_pricing
        # ts_date = asset_fama_macbeth_data.ts_date_pricing
        
        # idx = return_df.index.intersection(return_rf.index).intersection(return_index.index).intersection(ff_carhart_factors.index)
        idx = ts_date
        return_df = return_df.loc[idx, reduced_ticker_list.values()]
        return_rf = return_rf.loc[idx]
        return_index = return_index.loc[idx]
        
        return_df_ex = return_df.sub(return_rf.values, axis=0)
        return_index_ex = return_index.sub(return_rf.values, axis=0)        
 
        dates = return_df_ex.index
        
        lambda_list = []   # λ_t per period
        for t in dates:
            # t = dates[0]
            # print(f'on {t}')
            y_t = return_df_ex.loc[t, reduced_ticker_list.values()].dropna()
            if len(y_t) < 10:  # too few assets
                continue
            S = y_t.index
    
            # Factors (time-invariant per stock)
            X_t = betas.loc[S, :]
            X_t = X_t.loc[:, X_t.var(ddof=0) > 0]            
            m, p = X_t.shape[0], X_t.shape[1]
    
            if m > p+1: # regular
                # print('regular cross-sectional regression')
                X_t = sm.add_constant(X_t)
                res = sm.OLS(y_t, X_t).fit()
                λ_t = pd.Series(res.params, index=X_t.columns, name=t)
            else: # number of stocks less than number of factors/indicators
                # Ridge with intercept; coefficients are in original factor space
                # print('ridge cross-sectional regression')
                rc = RidgeCV(alphas=alphas, fit_intercept=True)
                rc.fit(X_t.values, y_t.values)
                λ_t = pd.Series([rc.intercept_, *rc.coef_],
                                index=['const']+list(X_t.columns), 
                                name=t)
            
            lambda_list.append(λ_t)
    
        # Stack to DataFrame: time × coefficients
        lambda_t = pd.DataFrame(lambda_list)
        lambda_mean = lambda_t.mean()
        
        # Newey–West t-stats across time
        def nw_t(series):
            y = series.dropna().values
            X = np.ones((len(y), 1))
            res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
            return res.params[0] / res.bse[0]
        lambda_tstats = lambda_t.apply(nw_t)
        
        lambda_dict = {"lambda_t": lambda_t,          # time-series of λ_t
                       "lambda_mean": lambda_mean,    # average risk premia
                       "lambda_tstat": lambda_tstats  # Newey–West t-stats of means
                       }
        
        lambda_summary = (pd.concat([lambda_mean, lambda_tstats], 
                             axis=1, keys=["λ̄_mean", "t_stat"]
                      ).sort_values("t_stat", ascending=False))
    
        lambda_summary.to_csv(file_dir+f'{index_name}_{method}_lambda_summary.csv')
        lambda_t.to_csv(file_dir+f'{index_name}_{method}_lambda_t.csv')
        
        return lambda_dict, lambda_summary, lambda_t

class asset_fama_macbeth_portfolio_class:
    def __init__(self, file_dir, T, T_momentum, index_name,
                 return_df, return_rf, #return_index,
                 betas, lambda_t):
        self.file_dir = file_dir
        self.T = T
        self.T_momentum = T_momentum
        self.index_name = index_name
        self.return_df = return_df
        self.return_rf = return_rf
        # self.return_index = return_index
        
        self.betas = betas
        self.lambda_t = lambda_t

        N = len(list(self.betas.index))
        self.risky_cap = round(1.0/N * 3, 3)
        
        self.shift = 1        
        self.weight_rf = 0.25

        # port_choice = 'return_weight'
        # portfolio weight is based on return
        self.weight_fm_portfolio_rw, self.return_fm_portfolio_rw = self.fama_macbeth_portfolio_weights('r')   
        # portfolio weight is based on return/std
        self.weight_fm_portfolio_rvw, self.return_fm_portfolio_rvw = self.fama_macbeth_portfolio_weights('rv')   
        
        # # portfolio weight is mean-variance gamma=2.0
        # self.weight_fm_portfolio_mv_gamma_2, self.return_fm_portfolio_mv_gamma_2 = self.fama_macbeth_portfolio_weights_mv(2)   
        # # portfolio weight is mean-variance gamma=5.0
        # self.weight_fm_portfolio_mv_gamma_5, self.return_fm_portfolio_mv_gamma_5 = self.fama_macbeth_portfolio_weights_mv(5)   
        # # tangent line (max sharpe daily)
        # self.weight_fm_portfolio_mv_tan, self.return_fm_portfolio_mv_tan = self.fama_macbeth_portfolio_weights_tan()   
        
        self.return_fm_portfolio = pd.concat([self.return_fm_portfolio_rw,
                                              self.return_fm_portfolio_rvw,
                                              # self.return_fm_portfolio_mv_gamma_2,
                                              # self.return_fm_portfolio_mv_gamma_5,
                                              # self.return_fm_portfolio_mv_tan
                                              ],
                                             axis=1)
                
    def fama_macbeth_portfolio_weights(self, port_choice):
        """
        Rolling daily FM portfolios (tradable).
          return_df_ex      : T×N DataFrame of daily *excess* returns (rows=dates, cols=tickers)
          betas    : N×K DataFrame of exposures (index=tickers, columns=K factors/indicators)
          lambda_t : T×K DataFrame of daily risk premia (same K columns as betas)
        """
        file_dir = self.file_dir
        T_momentum = self.T_momentum
        index_name = self.index_name
        return_df = self.return_df
        return_rf = self.return_rf
        # return_index = self.return_index        
        betas = self.betas
        lambda_t = self.lambda_t        
        risky_cap = self.risky_cap 
        shift = self.shift
        
        col_name_r = f'fama_macbeth_r_long_only_{index_name}'
        col_name_rv = f'fama_macbeth_rv_long_only_{index_name}'
        
        return_df_ex = return_df.sub(return_rf.values, axis=0)
        return_df_ex_std = return_df_ex.rolling(window=T_momentum).std()
                
        idx = return_df.index.intersection(return_rf.index).intersection(lambda_t.index)
        return_df = return_df.loc[idx]
        return_rf = return_rf.loc[idx]
        return_combined = pd.concat([return_df, return_rf], axis=1)
        return_df_ex = return_df_ex.loc[idx]
        return_df_ex_std = return_df_ex_std.loc[idx]
                
        tickers = betas.index.intersection(return_df_ex.columns)
        colsK = betas.columns.intersection(lambda_t.columns)
        
        betas = betas.loc[tickers, colsK]
        lambda_t = lambda_t.loc[:, colsK]        
        return_df_ex = return_df_ex.loc[:, tickers]
        return_df_ex_std = return_df_ex_std.loc[:, tickers]
        
        dates = return_df_ex.index.intersection(lambda_t.index)
        
        weights = []
        for t in dates:
            # t = dates[0]
            lam = lambda_t.loc[t]
            # expected return mu = beta * lambda
            mu_hat = (betas @ lam)#.dropna()    
            r_t = return_df_ex.loc[t, mu_hat.index]#.dropna()
            r_t_std = return_df_ex_std.loc[t, mu_hat.index]
            
            # common  = mu_hat.index.intersection(r_t.index)
            # mu_hat = mu_hat.loc[common]
            # r_t = r_t.loc[common]
                
            # construct long only portfolio based on mu_hat today
            w = mu_hat.copy(deep=True)
            w[w<0] = 0.0
           
            # if port_choice is return-volatility, w = positive mu_hat/std
            # otherwise, w is positive mu_hat
            if port_choice == 'rv':
                w = w
                w = w/r_t_std
            
            w.name = t
            w = w.to_frame().T 
            w = w.divide(w.sum(axis=1), axis=0)
            w = w.clip(0, risky_cap)
            w[return_rf.columns[0]] = 1 - w.sum(axis=1)
            
            weights.append(w)
           
        weights = pd.concat(weights, axis=0)#.sort_index
        return_portfolio = weights.shift(shift).mul(return_combined).sum(axis=1)
        if port_choice == 'r':
            return_portfolio.name = col_name_r
        elif port_choice == 'rv':
            return_portfolio.name = col_name_rv

        weights.to_csv(file_dir+f'{index_name}_fama_macbeth_{port_choice}_long_only_weight.csv')
        return_portfolio.to_csv(file_dir+f'{index_name}_fama_macbeth_{port_choice}_long_only_return.csv')

        return weights, return_portfolio

    def fama_macbeth_portfolio_weights_mv(self, gamma=2):
        """
        Rolling daily FM portfolios (tradable).
          return_df_ex      : T×N DataFrame of daily *excess* returns (rows=dates, cols=tickers)
          betas    : N×K DataFrame of exposures (index=tickers, columns=K factors/indicators)
          lambda_t : T×K DataFrame of daily risk premia (same K columns as betas)
        """
        file_dir = self.file_dir
        T_momentum = self.T_momentum
        index_name = self.index_name
        return_df = self.return_df
        return_rf = self.return_rf
        # return_index = self.return_index        
        betas = self.betas
        lambda_t = self.lambda_t        
        # max_weight = self.max_weight  
        shift = self.shift
        
        col_name = f'fama_macbeth_mv_long_only_gamma_{gamma}_{index_name}'
        
        return_combined = pd.concat([return_df, return_rf], axis=1)        
        return_df_ex = return_df.sub(return_rf.values, axis=0) 
        
        return_combined_ex = pd.concat([return_df_ex, return_rf], axis=1)
                
        tickers = betas.index.intersection(return_df_ex.columns)
        colsK = betas.columns.intersection(lambda_t.columns)        
        betas = betas.loc[tickers, colsK]
        lambda_t = lambda_t.loc[:, colsK]  
        # return_combined_ex = return_combined_ex.loc[:, tickers]
        
        # expected return mu = beta * lambda
        mu_T = lambda_t @ betas.T
        # and add rf_t 
        mu_T = pd.concat([mu_T, return_rf], axis=1).dropna(how='any')
        
        dates = return_combined_ex.index.intersection(lambda_t.index)
        return_combined = return_combined.loc[dates]
        
        weights = []
        for t in dates:
            # t = dates[0]
            mu_t = mu_T.loc[t]
            sigma_t = return_combined_ex.loc[:t, mu_t.index].tail(T_momentum).cov()
            w_t = self.solve_mv(t, mu_t, sigma_t, gamma)
            weights.append(w_t)
           
        weights = pd.concat(weights, axis=1).T#.sort_index
        
        return_portfolio = weights.shift(shift).mul(return_combined).sum(axis=1)
        return_portfolio.name = col_name

        weights.to_csv(file_dir+f'{index_name}_fama_macbeth_mv_gamma_{gamma}_long_only_weight.csv')
        return_portfolio.to_csv(file_dir+f'{index_name}_fama_macbeth_mv_gamma_{gamma}_long_only_return.csv')

        return weights, return_portfolio
    
    def solve_mv(self, t, mu_t, sigma_t, gamma):
        tickers = list(mu_t.index)
        N = len(tickers)
        risky_cap = round(1.0/(N-1) * 3, 3)
        rf_cap = 1.0
    
        # objective: 0.5*gamma * w'Σw - μ'w  (convex)
        def objective(w):
            return 0.5 * gamma * w @ sigma_t @ w - mu_t @ w
    
        bounds = [(0.0, risky_cap)] * (N-1) + [(0.0, rf_cap)]  # length N+1
        cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}  # fully invested
        
        # start at all risk-free (always feasible)
        w0 = np.zeros(N); w0[-1] = 1.0  
        res = minimize(objective, w0, method='SLSQP', 
                       bounds=bounds, constraints=cons)
        
        w_t = w0 if not res.success else res.x
        w_t = w_t.round(3) #np.ceil(w_t * 100) / 100 # two decimal places
        w_t = pd.Series(w_t, index=tickers)
        w_t.name = t
        
        return w_t

    def fama_macbeth_portfolio_weights_tan(self):
        """
        Long-only max-Sharpe (tangency) portfolio over *risky assets only*.
          mu       : pd.Series length N   (expected TOTAL returns for the period)
          Sigma    : pd.DataFrame NxN     (covariance, same order as mu.index)
          rf       : float                (risk-free return for the period)
          stock_cap: float                (upper bound per stock, e.g. 0.10)
    
        Returns: pd.Series of weights (sum=1, w>=0).
        """
        file_dir = self.file_dir
        T_momentum = self.T_momentum
        index_name = self.index_name
        return_df = self.return_df
        return_rf = self.return_rf
        weight_rf = self.weight_rf        
        betas = self.betas
        lambda_t = self.lambda_t        
        # max_weight = self.max_weight  
        shift = self.shift
        weight_rf = self.weight_rf
        
        col_name = f'fama_macbeth_mv_tan_{index_name}'
        
        return_combined = pd.concat([return_df, return_rf], axis=1)  
        return_df_ex = return_df.sub(return_rf.values, axis=0) 
                        
        tickers = betas.index.intersection(return_df_ex.columns)
        colsK = betas.columns.intersection(lambda_t.columns)        
        betas = betas.loc[tickers, colsK]
        lambda_t = lambda_t.loc[:, colsK]  
        
        # expected return mu = beta * lambda
        mu_T = lambda_t @ betas.T       
        dates = return_df_ex.index.intersection(lambda_t.index)
        return_combined = return_combined.loc[dates]
        
        weights = []
        for t in dates:
            # t = dates[0]
            mu_t = mu_T.loc[t]
            sigma_t = return_df_ex.loc[:t, mu_t.index].tail(T_momentum).cov()
            w_t = self.solve_mv_tan(t, mu_t, sigma_t, weight_rf, return_rf.columns[0])
            weights.append(w_t)
            
        weights = pd.concat(weights, axis=1).T#.sort_index
        
        return_portfolio = weights.shift(shift).mul(return_combined).sum(axis=1)
        return_portfolio.name = col_name
        
        weights.to_csv(file_dir+f'{index_name}_fama_macbeth_mv_tan_long_only_weight.csv')
        return_portfolio.to_csv(file_dir+f'{index_name}_fama_macbeth_mv_tan_long_only_return.csv')

        return weights, return_portfolio        
    
    def solve_mv_tan(self, t, mu_t, sigma_t, weight_rf, col_rf):        
        tickers = list(mu_t.index)
        N = len(tickers)
        risky_cap = round(1.0/N * 3, 3)
        
        # Objective: NEGATIVE Sharpe (we will minimize it)
        def neg_sharpe(w):
            # mu_ex = m - rf            # excess mean vector
            num = w @ mu_t
            den = np.sqrt(max(w @ sigma_t @ w, 1e-12))
            return -(num / den)
    
        # Bounds: long-only, cap per stock
        bounds = [(0.0, risky_cap)] * N
        # Constraints: sum w = 1
        cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
    
        # Start from equal-weights (clipped to cap)
        w0 = np.full(N, 1.0 / N)    
        res = minimize(neg_sharpe, w0, method='SLSQP', 
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 500, 'ftol': 1e-9})
        w_t = w0 if not res.success else res.x   
        w_t = w_t/sum(w_t) * (1 - weight_rf)
        w_t = np.append(w_t, weight_rf)     
        w_t = w_t.round(3) #np.ceil(w_t * 100) / 100 # two decimal places
        tickers.append(col_rf)        
        w_t = pd.Series(w_t, index=tickers)
        w_t.name = t
        
        return w_t
#%%    
    
    
    
    
    