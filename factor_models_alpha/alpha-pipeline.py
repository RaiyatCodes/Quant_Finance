#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECON367B Fall 2025 factor/indicator, test and portfolio, fama-macbeth excercise
@author: raiyat 
"""

# quant_pipeline_all_in_one_plus_FMB
# --------------------------------------------------------------------
# includes:
#   - Daily indicator generation (TA-Lib curated + liquidity + SO-based)
#   - Alphalens testing (ICs & long–short)
#   - Alpha portfolio construction & backtest

# pip install TA-Lib 
# pip install alphalens-reloaded

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import talib
import alphalens as al
from pandas import Timedelta

class asset_indicator_data_class:
    def __init__(self, file_dir, T, T_momentum, index_name, ticker_list, 
                 asset_return_df, asset_close_df,
                 asset_ticker_df_dict, asset_index_df_dict, 
                 asset_fundamental_data_dict, ff_carhart_factors):
        self.file_dir = file_dir# = work_dir
        self.T = T
        self.T_momentum = T_momentum        
        self.index_name = index_name
        self.ticker_list = ticker_list
        self.return_df = asset_return_df
        self.close_df = asset_close_df
        self.ff_carhart_factors = ff_carhart_factors                
        self.ticker_df_dict = asset_ticker_df_dict
        self.index_df_dict = asset_index_df_dict
        
        self.reduced_ticker_list = {k: v for k, v in ticker_list.items() if v in self.return_df.columns}
        self.ticker_fundamental_dict = {self.reduced_ticker_list[k]: v for k, v in asset_fundamental_data_dict.items()}
                     
        self.ticker_indicator_dict, self.ticker_all_indicator_dict, self.parameter_all_indicator_dict  = self.get_ticker_indicators()
        
        periods = (1, int(T_momentum/2), T_momentum, 2*T_momentum)            
        # -----------------------------------------------
        # Alphalens evaluation for all indicators
        # -----------------------------------------------
        results_summary = {}
        results_ic_ts = {}
        results_ls_ts = {}
        results_ic_mean_by_stock = {}    
        results_ic_t_by_stock = {}
        # parameter_all_indicator_dict = asset_indicator_data.parameter_all_indicator_dict
        for indicator, factor_df in self.parameter_all_indicator_dict.items():
            print(f"Evaluating indicator: {indicator}")       
            # indicator = 'AD'
            # factor_df = parameter_all_indicator_dict[indicator]
            (ic_mean, ic_t, ls_mean, ls_t,
             ic_ts, ls_ts, fd,
             per_stock_ic, per_stock_ic_t
            ) = self.evaluate(indicator, factor_df, self.close_df, periods)
            
            ic_ts.index = ic_ts.index.date
            ls_ts.index = ls_ts.index.date
            # ---- summary (average across horizons) ----
            summary = pd.DataFrame({
                "IC_mean": ic_mean,
                "IC_t": ic_t,
                "LS_mean": ls_mean,
                "LS_t": ls_t
            }).T  # horizons as columns
            # ic and ls values in mean
            results_summary[indicator] = summary
            
            # ic and ts values in time series
            results_ic_ts[indicator] = ic_ts
            results_ic_ts[indicator].to_csv(file_dir+index_name+'_'+indicator+'_ic_ts.csv')           
            results_ls_ts[indicator] = ls_ts
            results_ls_ts[indicator].to_csv(file_dir+index_name+'_'+indicator+'_ls_ts.csv')
                  
            # ---- per-stock results ----
            results_ic_mean_by_stock[indicator] = per_stock_ic
            results_ic_mean_by_stock[indicator].to_csv(file_dir+index_name+'_'+indicator+'_ic_mean_by_stock_ts.csv')
            results_ic_t_by_stock[indicator] = per_stock_ic_t
            results_ic_t_by_stock[indicator].to_csv(file_dir+index_name+'_'+indicator+'_ic_t_by_stock_ts.csv')
        
        # ---------- Collect results ----------
        summary_all = pd.concat(results_summary, axis=0)
        summary_all.index = summary_all.index.set_names(["Indicator", "Metric"])
        summary_all = summary_all.round(4)
        ic_mean_all = summary_all[summary_all.index.get_level_values(1)=='IC_mean']
        ls_mean_all = summary_all[summary_all.index.get_level_values(1)=='LS_mean']
        
        summary_all.to_csv(file_dir+index_name+'_all_indicator_summary.csv')
        ic_mean_all.to_csv(file_dir+index_name+'_all_indicator_ic_mean.csv')
        ls_mean_all.to_csv(file_dir+index_name+'_all_indicator_ls_mean.csv')
        
        ### horizon (days): stocks x indicators
        panel = pd.concat(results_ic_mean_by_stock, axis=1)
        fields = panel.columns.get_level_values(1).unique()             
        results_ic_mean_by_stock_indicator = {}
        for f in fields:
            print(f)
            results_ic_mean_by_stock_indicator[f] = panel.xs(f, axis=1, level=1)
            results_ic_mean_by_stock_indicator[f].to_csv(file_dir+index_name+'_'+str(f)+'D_ic_mean_by_stock_indicator_horizon_ts.csv')

        panel = pd.concat(results_ic_t_by_stock, axis=1)
        fields = panel.columns.get_level_values(1).unique()             
        results_ic_t_by_stock_indicator = {}
        for f in fields:
            results_ic_t_by_stock_indicator[f] = panel.xs(f, axis=1, level=1)
            results_ic_t_by_stock_indicator[f].to_csv(file_dir+index_name+'_'+str(f)+'D_ic_t_by_stock_indicator_horizen_ts.csv')
    
        ### stock: indicator x horizons
        dict_mean = {}       
        for indicator, df in results_ic_mean_by_stock.items():
            for stock in df.index:
                if stock not in dict_mean:
                    dict_mean[stock] = pd.DataFrame(columns=df.columns)
                dict_mean[stock].loc[indicator] = df.loc[stock]

        # Save each stock's dataframe
        for ticker, df in dict_mean.items():
            df.to_csv(file_dir+(index_name+'_'+ticker+'_ic_mean_by_indicator_horizon_ts.csv').replace('/','_'))

        dict_t = {}       
        for indicator, df in results_ic_t_by_stock.items():
            for stock in df.index:
                if stock not in dict_t:
                    dict_t[stock] = pd.DataFrame(columns=df.columns)
                dict_t[stock].loc[indicator] = df.loc[stock]  
                
        for ticker, df in dict_t.items():
            df.to_csv(file_dir+(index_name+'_'+ticker+'_ic_t_by_indicator_horizon_ts.csv').replace('/','_'))
    
        self.indicator_summary_dict = results_summary
        self.indicator_ic_ts_dict = results_ic_ts
        self.indicator_ls_ts_dict = results_ls_ts
        self.indicator_ic_mean_by_stocks_dict = results_ic_mean_by_stock
        self.indicator_ic_t_by_stocks_dict = results_ic_t_by_stock
        
        self.horizon_ic_mean_stocks_indicators_dict = results_ic_mean_by_stock_indicator
        self.horizon_ic_t_stocks_indicators_dict = results_ic_t_by_stock_indicator
        
        self.stock_ic_mean_indicators_horizon_dict = dict_mean
        self.stock_ic_t_indicators_horizon_dict = dict_t
               
        self.indicator_ic_summary = summary_all
        self.indicator_ic_mean = ic_mean_all
        self.indicator_ls_mean = ls_mean_all
        
        horizons = self.indicator_ic_summary.columns
        # horizons = asset_indicator_data.indicator_ic_summary.columns
        ic_by_horizon = {}       
        for h in horizons:
            # Build a new DataFrame for this horizon: rows=days, cols=indicators
            cols = {ind: df[h] for ind, df in self.indicator_ic_ts_dict.items() if h in df.columns}
            ic_by_horizon[h] = pd.DataFrame(cols).sort_index()  # 749 x 44 (days x indicators)
            ic_by_horizon[h].to_csv(file_dir+index_name+'_'+h+'_ic_t_ts.csv')
        self.horizon_ic_t_ts_dict = ic_by_horizon
    
        _ic_bar_chart = self._plot_bar_chart(self.indicator_ic_mean)
        _ls_bar_chart = self._plot_bar_chart(self.indicator_ls_mean)
        
        for indicator, df in self.indicator_ic_ts_dict.items():
            _ic_line_plot = self._plot_line_chart(indicator, df, self.indicator_ic_mean.loc[indicator])
              
        for indicator, df in self.indicator_ic_ts_dict.items():
            _ic_scattor_plot = self._ic_scattor_plot(indicator, df, self.indicator_ic_mean.loc[indicator]) 
            
        # self.w_ic = self.compute_ic_weights(horizon="1D")
        
    # def compute_ic_weights(self, horizon="1D", var_floor=1e-6):
    #     """
    #     Compute factor combination weights:
    #         w_f ∝ mean(IC_f) / var(IC_f)   
    #     Parameters
    #     ----------
    #     ic_ts_dict : dict
    #         {factor_name: DataFrame(dates × horizons)} for each factor.
    #     horizon : str
    #         Which horizon column to use (e.g. '1D', '5D', etc.)
    #     var_floor : float
    #         Lower bound to prevent divide-by-zero.
    
    #     Returns
    #     -------
    #     weights : dict
    #         Normalized to sum(|w|) = 1.
    #     """   
    #     ic_ts_dict = self.indicator_ic_ts_dict
    #     # ic_ts_dict = asset_indicator_data.indicator_ic_ts_dict
                
    #     raw_w = {}
    #     for indicator, ic_df in ic_ts_dict.items():   
    #         # indicator = 'AD'
    #         # ic_df = ic_ts_dict[indicator]           
    #         s = ic_df[horizon].dropna()
    #         if len(s) < 5:
    #             continue
    #         mean_ic = float(s.mean())
    #         var_ic = float(s.var(ddof=1))
    #         raw_w[indicator] = (mean_ic / max(var_ic, var_floor)) if mean_ic>0 else 0.0
    
    #     # Normalize so sum of absolute weights = 1
    #     total = sum(abs(v) for v in raw_w.values()) or 1.0
    #     weights = {k: v / total for k, v in raw_w.items()}
        
    #     return weights
    
    def _ic_scattor_plot(self, indicator, df, ic_mean):   
        file_dir = self.file_dir
        index_name = self.index_name

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.ravel()
        for i, c in enumerate(ic_mean.columns):
            # print(i, c)
            ax = axes[i]
            ax.scatter(df.index, df[c], s=10, alpha=0.7)
            ax.axhline(ic_mean[c].values[0], ls="--", color="red")
            ax.set_title(f"{indicator} – {c}")
            ax.grid(True, ls="--", alpha=0.4)

        for j in range(len(ic_mean.columns), 4):
            axes[j].axis("off")

        plt.tight_layout()
        plot_name = f'{index_name}_{indicator}_ic_ts_vs_ic_mean'
        fig.suptitle(plot_name, fontsize=14)        
        plt.savefig(file_dir+plot_name+'.png', dpi=300, bbox_inches="tight")
        plt.show()
        
        return None
               
    def _plot_line_chart(self, indicator, df, ic_mean):
        file_dir = self.file_dir
        index_name = self.index_name
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.ravel()
        for i, c in enumerate(df.columns):
            # print(i, c)
            ax = axes[i]
            ax.plot(df.index, df[c], lw=1.0, alpha=0.7)
            ax.axhline(ic_mean[c].values[0], ls="--", color="red")
            ax.set_title(f"{indicator} – {c}")
            ax.grid(True, ls="--", alpha=0.4)

        for j in range(len(df.columns), 4):
            axes[j].axis("off")
            
        # ✅ Force ticks/labels visible on ALL subplots
        for ax in axes:
            ax.tick_params(which="both", labelbottom=True, labelleft=True)
            for lbl in ax.get_xticklabels() + ax.get_yticklabels():
                lbl.set_visible(True)
       
        plt.tight_layout()
        plot_name = f'{index_name}_{indicator}_ic_ts'
        fig.suptitle(plot_name, fontsize=14)       
        plt.savefig(file_dir+plot_name+'.png', dpi=300, bbox_inches="tight")
        plt.show()
        
        return None
    
    def _plot_bar_chart(self, df):
        file_dir = self.file_dir
        index_name = self.index_name
        # df = asset_indicator_data.indicator_ic_mean
        # indicator_ls_mean = self.indicator_ls_mean
        """
        Plot four pie charts (1D, 5D, 10D, 20D) for each indicator.
        The DataFrame has MultiIndex (indicator, Metric) but Metric is ignored.
        """
        
        metric = df.index.get_level_values(1)[0]
        # # --- collapse multiindex over indicator ---
        if isinstance(df.index, pd.MultiIndex):
            df = df.groupby(level=0).mean(numeric_only=True)
    
        # --- pick the  columns ---
        cols = df.columns
        
        # --- plot ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
    
        for i, col in enumerate(cols):
            ax = axes[i]
            df[col].plot(kind='bar', ax=ax)
            ax.set_title(f"{col} IC by Indicator", fontsize=12)
            ax.set_xlabel("Indicator")
            ax.set_ylabel("IC Mean")
            ax.grid(axis='y', linestyle='--', alpha=0.6)
    
        # hide any unused subplots (if fewer than 4)
        for j in range(len(cols), 4):
            axes[j].axis('off')
    
        plot_name = f'{index_name}_{metric}_by_horizon'
        plt.suptitle(plot_name, fontsize=14)
        plt.tight_layout()       
        # --- save to file ---
        plt.savefig(file_dir+plot_name, dpi=300, bbox_inches='tight')
        plt.show()
        
        return None
    
    def get_ticker_indicators(self):
        file_dir = self.file_dir
        T = self.T
        T_momentum = self.T_momentum
        index_name = self.index_name
        return_df = self.return_df # return data for all qualified stocks in index
        ticker_df_dict = self.ticker_df_dict
        index_df_dict = self.index_df_dict
        ff_carhart_factors = self.ff_carhart_factors
        ticker_fundamental_dict = self.ticker_fundamental_dict
        reduced_ticker_list = self.reduced_ticker_list
        
        freq = lookback = T_momentum
        return_df = return_df.loc[ff_carhart_factors.index]        
        # rolling window is of T, 252 days
        dates = return_df.index[T_momentum-1:]
    
        ticker_indicator_dict = {}
        # loop through the assets
        X = index_df_dict[index_name].loc[ff_carhart_factors.index]
        for ticker in ticker_df_dict.keys():
            print(ticker)
            Y = ticker_df_dict[ticker].loc[ff_carhart_factors.index] 
            Z = ticker_fundamental_dict[ticker].loc[ff_carhart_factors.index]
            ticker_indicator_dict[ticker] = factor_indicator_tester_class(X, Y, Z, T_momentum, ticker)
                 
        # ticker_indicator_dict = asset_indicator_data.ticker_indicator_dict        
        ticker_all_indicator_dict = {}
        for ticker in ticker_indicator_dict.keys():
            # ticker = 'Nvidia'
            ticker_all_indicator_dict[ticker] = ticker_indicator_dict[ticker].all_factors_df
            ticker_all_indicator_dict[ticker].to_csv(file_dir+(index_name+'_'+ticker+'_all_indicators.csv').replace('/','_'))
        
        panel = pd.concat(ticker_all_indicator_dict, axis=1)
        fields = panel.columns.get_level_values(1).unique()       
        # return_dates_dict = {f: panel.xs(f, axis=1, level=1) for f in fields}        
        parameter_all_indicator_dict = {}
        for f in fields:
            print(f)
            parameter_all_indicator_dict[f] = panel.xs(f, axis=1, level=1)
            parameter_all_indicator_dict[f].to_csv(file_dir+index_name+'_'+f+'_all_tickers.csv')

        return ticker_indicator_dict, ticker_all_indicator_dict, parameter_all_indicator_dict
    
    def evaluate(self, indicator, factor_df, prices_df, periods=(1,5,10,20)):
        # prices_df = asset_indicator_data.close_df
        """
        Evaluate a single TA-Lib indicator (factor) across multiple stocks with Alphalens.
    
        factor_df : pd.DataFrame
            rows = dates, columns = tickers, values = factor values
        prices_df : pd.DataFrame
            rows = dates, columns = tickers, values = close prices
        periods : tuple
            forward-return horizons for Alphalens
        """
        # ---------- Clean alignment ----------
        factor_df = factor_df.copy()
        factor_df.index  = pd.to_datetime(factor_df.index).tz_localize(None)
        prices_df = prices_df.copy()
        prices_df.index = pd.to_datetime(prices_df.index).tz_localize(None)
    
        # align dates and tickers
        common_dates   = factor_df.index.intersection(prices_df.index)
        common_tickers = factor_df.columns.intersection(prices_df.columns)
        factor_df = factor_df.loc[common_dates, common_tickers]
        prices_df = prices_df.loc[common_dates, common_tickers]
    
        # ---------- Prepare factor for Alphalens ----------
        f = factor_df.stack().rename("factor").sort_index()
    
        # ---------- Run Alphalens ----------
        fd = al.utils.get_clean_factor_and_forward_returns(
            f, prices_df, quantiles=5, periods=periods, max_loss=1.0
        )
    
        # ---------- Portfolio-level metrics (cross-sectional, per horizon) ----------
        ic_ts = al.performance.factor_information_coefficient(fd)  # time series per horizon
        ls_ts = al.performance.factor_returns(fd)                  # long–short per horizon
    
        ic_mean = ic_ts.mean()
        ic_std  = ic_ts.std(ddof=1)
        ic_t    = ic_mean / (ic_std / np.sqrt(ic_ts.count()))
    
        ls_mean = ls_ts.mean()
        ls_std  = ls_ts.std(ddof=1)
        ls_t    = ls_mean / (ls_std / np.sqrt(ls_ts.count()))
    
        # ---------- Per-stock ICs (time-series Spearman corr of factor vs fwd returns) ----------
        # Build a per-horizon table of per-stock correlations
        per_stock_ic_dict = {}
        per_stock_ic_t_dict = {}
    
        for p in periods:
            # forward return column name used by Alphalens
            col = f"{p}D"
            if col not in fd.columns:
                # fallbacks (robust to different AL versions / settings)
                fallbacks = [f"{p}d", f"{p}", f"period_{p}", f"forward_returns_{p}"]
                for alt in fallbacks:
                    if alt in fd.columns:
                        col = alt
                        break
                else:
                    raise KeyError(f"Could not find forward return column for period {p}. "
                                   f"Available: {list(fd.columns)}")
        
            tmp = fd[['factor', col]].dropna()
        
            # Spearman per-asset IC across time
            ic_by_asset = (
                tmp.groupby(level='asset')
                   .apply(lambda x: x['factor'].corr(x[col], method='spearman'))
            )
        
            # t-stat for correlation: t = r * sqrt((n-2)/(1-r^2))
            n_by_asset = tmp.groupby(level='asset')[col].count()
            r = ic_by_asset
            n = n_by_asset.reindex(r.index).fillna(0)
            with np.errstate(divide='ignore', invalid='ignore'):
                t_vals = r * np.sqrt((n - 2).clip(lower=0) / (1 - r**2).replace(0, np.nan))
        
            per_stock_ic_dict[p] = r
            per_stock_ic_t_dict[p] = t_vals
            
        # Assemble as DataFrames: rows=assets, cols=horizons
        per_stock_ic   = pd.DataFrame(per_stock_ic_dict).sort_index()
        per_stock_ic_t = pd.DataFrame(per_stock_ic_t_dict).sort_index()
    
        return (
            ic_mean, ic_t, ls_mean, ls_t,  # portfolio-level summaries (per horizon)
            ic_ts, ls_ts,                  # portfolio-level time series (per horizon)
            fd,                            # cleaned factor/forward-returns panel
            per_stock_ic, per_stock_ic_t   # NEW: per-stock IC and per-stock IC t-stats per horizon
        )

# =========================================================
# 1) Indicator generation + Alphalens testing
# =========================================================
class factor_indicator_tester_class:
    def __init__(self, X, Y, Z, T_momentum, ticker):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.T_momentum = T_momentum
        self.ticker = ticker
        self.close = Y['Close']
        self.high = Y['High']
        self.low = Y['Low']
        self.volume = Y['Volume']
        self.market = X['Close'] # market_close # optional Series aligned to dates
        self.shares = Z['shares'] # optional DF
        self.float_shares = Z['shares'] # optional DF        
        self.shift = 1
        
        self.talib_factors = self.build_talib_factors()
        self.liq_factors = self.build_liquidity_factors()
        self.so_factors = self.build_so_based_factors()
        self.all_factors = all_factors = {**self.talib_factors, **self.liq_factors, **self.so_factors}
        self.all_factors_df = pd.concat(self.all_factors, axis=1)
        self.all_factors_df.columns = self.all_factors_df.columns.get_level_values(0)

    # ---------- helpers ----------
    def _zscore_xsec(self, df):
        m = df.mean()
        s = df.std()
        return df.sub(m).div(s)

    # ---------- curated TA-Lib daily indicators ----------
    def build_talib_factors(self):
        C, H, L, V = self.close, self.high, self.low, self.volume
        M = self.market
        T_momentum = self.T_momentum
        shift = self.shift
        # C, H, L, V = close, high, low, volume
        
        F = {}
        # Trend / momentum
        # rsi
        F["RSI_T"] = pd.DataFrame(talib.RSI(C, T_momentum), columns=['RSI_T'])
        # macd
        _, _, h = talib.MACD(C.values)
        F["MACD_hist"] = pd.DataFrame(h, index=C.index, columns=['MACD_hist'])
        F["ROC_2T"] = pd.DataFrame(talib.ROC(C, 2*T_momentum), columns=['ROC_2T'])
        F["TRIX"] = pd.DataFrame(talib.TRIX(C, 3*T_momentum), columns=['TRIX'])
        F["PPO"] = pd.DataFrame(talib.PPO(C, T_momentum, 2*T_momentum, 0), columns=['PPO'])
        F["APO"] = pd.DataFrame(talib.APO(C, T_momentum, 2*T_momentum, 0), columns=['APO'])

        # Stochastics (needs H/L)        
        stk, std = talib.STOCH(H.values, L.values, C.values)
        fstk, fstd = talib.STOCHF(H.values, L.values, C.values)       
        F["STOCH_K"] = pd.DataFrame(stk, index=C.index, columns=['STOCH_K'])
        F["STOCH_D"] = pd.DataFrame(std, index=C.index, columns=['STOCH_D'])
        F["STOCHF_K"] = pd.DataFrame(fstk, index=C.index, columns=['STOCHF_K'])
        F["STOCHF_D"] = pd.DataFrame(fstd, index=C.index, columns=['STOCHF_D'])

        # Regime / strength
        F["ADX_T"] = pd.DataFrame(talib.ADX(H, L, C, T_momentum), columns=['ADX_T'])
        
        adn, aup = talib.AROON(H.values, L.values, T_momentum)        
        F["AROON_UP"] = pd.DataFrame(aup, index=C.index, columns=['AROON_UP'])
        F["AROON_DOWN"] = pd.DataFrame(adn, index=C.index, columns=['AROON_DOWN'])
        F["AROONOSC"]   = pd.DataFrame(talib.AROONOSC(H.values, L.values, T_momentum),
                                       index=C.index, columns=['AROONOSC'])        
        F["HT_TRENDMODE"] = pd.DataFrame(talib.HT_TRENDMODE(C), columns=['HT_TRENDMODE'])
                                         
        # Volatility / risk
        F["ATR_T"] = pd.DataFrame(talib.ATR(H, L, C, T_momentum),columns=['ATR_T'])
        F["NATR_T"]  = pd.DataFrame(talib.NATR(H, L, C, T_momentum),columns=['NATR_T'])
        F["TRANGE"]  = pd.DataFrame(talib.TRANGE(H, L, C), columns=['TRANGE'])
        F["STDDEV_2T"] = pd.DataFrame(talib.STDDEV(C, 2*T_momentum, 1), columns=['STDDEV_2T'])

        # Bands / position
        up, mid, lo = talib.BBANDS(C.values, 2*T_momentum)
        up = pd.Series(up, index=C.index)
        lo = pd.Series(lo, index=C.index)
        denom = (up - lo).replace(0, np.nan)
        bb_pctB  = (C - lo) / denom
        bb_width = up - lo        
        F["BB_pctB"] = pd.DataFrame(bb_pctB, columns=['BB_pctB'])
        F["BB_width"] = pd.DataFrame(bb_width, columns=['BB_width'])
        F["KAMA_T"] = pd.DataFrame(talib.KAMA(C, T_momentum), columns=['KAMA_T'])

        # Price transforms
        F["TYPPRICE"] = pd.DataFrame(talib.TYPPRICE(H, L, C), columns=['TYPPRICE'])
        F["WCLPRICE"] = pd.DataFrame(talib.WCLPRICE(H, L, C), columns=['WCLPRICE'])

        # Volume/flow
        F["OBV"] = pd.DataFrame(talib.OBV(C, V), columns=['OBV'])
        F["AD"] = pd.DataFrame(talib.AD(H, L, C, V), columns=['AD'])
        F["ADOSC"] = pd.DataFrame(talib.ADOSC(H, L, C, V, int(T_momentum/3), T_momentum), columns=['ADOSC'])
        F["MFI_T"] = pd.DataFrame(talib.MFI(H, L, C, V, T_momentum), columns=['MFI_T'])

        # Simple statistical
        F["LINREG_SLOPE_2T"] = pd.DataFrame(talib.LINEARREG_SLOPE(C, 2*T_momentum), columns=['LINREG_SLOPE_2T'])
        F["LINREG_ANGLE_2T"] = pd.DataFrame(talib.LINEARREG_ANGLE(C, 2*T_momentum), columns=['LINREG_ANGLE_2T'])

        # Optional: BETA/CORREL vs market (63)        
        F["BETA_2T_MKT"] = pd.DataFrame(talib.BETA(C, M, 2*T_momentum), columns=['BETA_2T_MKT'])
        F["CORR_2T_MKT"] = pd.DataFrame(talib.CORREL(C, M, 2*T_momentum), columns=['CORR_2T_MKT'])

        # z-score + lag(1)
        out = {}
        for k, df in F.items():          
            # m = df.mean()
            # s = df.std()
            # z = df.sub(m).div(s)
            z = self._zscore_xsec(df)
            out[k] = z.shift(shift)
        
        return out

    # ---------- liquidity (no SO) ----------
    def build_liquidity_factors(self):
        C, V = self.close, self.volume
        T_momentum = self.T_momentum
        shift = self.shift
        # C, V = close,  volume
        
        F = {}
        dollar_vol = (C * V).replace(0, np.nan)
        ret = C.pct_change()

        F["LOG_DOLLAR_VOL"] = pd.DataFrame(np.log(dollar_vol), columns=['LOG_DOLLAR_VOL'])
        F["ADV_2T"] = pd.DataFrame(dollar_vol.rolling(2*T_momentum, min_periods=T_momentum).mean(), 
                                   columns=['ADV_2T'])
        F["RVOL_2T"] = (V/(V.rolling(2*T_momentum, min_periods=T_momentum).mean())).to_frame()
        F["RVOL_2T"].columns = pd.Index(['RVOL_2T'])

        illiq_d = (ret.abs() / dollar_vol).replace([np.inf, -np.inf], np.nan)
        F["AMIHUD_2T"] = pd.DataFrame(illiq_d.rolling(2*T_momentum, min_periods=T_momentum).mean(),
                                      columns=['AMIHUD_2T'])

        out = {}
        for k, df in F.items():
            z = self._zscore_xsec(df)
            out[k] = z.shift(shift)
        return out

    # ---------- shares-outstanding based ----------
    def build_so_based_factors(self):
        C, V, SO, FSO = self.close, self.volume, self.shares, self.float_shares
        T_momentum = self.T_momentum
        shift = self.shift
        # C, V, SO, FSO = close, volume, shares, float_shares

        out = {}

        mcap = (C * SO).replace(0, np.nan)
        out["LOG_MCAP"] = self._zscore_xsec(np.log(mcap)).shift(shift).to_frame()
        out["LOG_MCAP"].columns = ['LOG_MCAP']

        turnover = (V / SO.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["TURNOVER"] = self._zscore_xsec(turnover).shift(shift).to_frame()
        out['TURNOVER'].columns = ['TURNOVER']
        out["RTURN_2T"] = self._zscore_xsec(turnover / turnover.rolling(2*T_momentum, min_periods=T_momentum).mean()).shift(shift).to_frame()
        out["RTURN_2T"].columns = ['RTURN_2T']
        out["CHURN_2T"] = self._zscore_xsec(turnover.rolling(2*T_momentum, min_periods=T_momentum).sum()).shift(shift).to_frame()
        out['CHURN_2T'].columns = ['CHURN_2T']

        issuance = np.log(SO).diff()
        out["ISSUANCE_2T"] = self._zscore_xsec(issuance.rolling(2*T_momentum, min_periods=T_momentum).mean()).shift(shift).to_frame()
        out["ISSUANCE_2T"].columns = ["ISSUANCE_2T"]

        std20 = C.pct_change().rolling(2*T_momentum, min_periods=T_momentum).std()
        out["TURNOVER_PER_VOL"] = self._zscore_xsec(turnover / std20.replace(0, np.nan)).shift(shift).to_frame()
        out["TURNOVER_PER_VOL"].columns = ["TURNOVER_PER_VOL"]

        fturn = (V / FSO.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["FLOAT_TURNOVER"] = self._zscore_xsec(fturn).shift(shift).to_frame()
        out["FLOAT_TURNOVER"].columns = ["FLOAT_TURNOVER"]        
        out["FCHURN20"] = self._zscore_xsec(fturn.rolling(2*T_momentum, min_periods=T_momentum).sum()).shift(shift).to_frame()
        out["FCHURN20"].columns = ["FCHURN20"]

        return out

# =========================================================
# 2) Alpha portfolio from tested factors
# =========================================================
# =========================================================
# Alpha Portfolio (1-day horizon, auto-run version)
# =========================================================
class asset_alpha_portfolio_data_class:
    def __init__(self, file_dir, T, T_momentum, index_name, ticker_list,
                 returns_df, ff_carhart_factors, factors, 
                 factor_ics, max_weight=0.05, cost_bps=2.0):
        """
        Automatically builds and backtests an alpha portfolio upon creation.

        factors_dict: {factor_name: DataFrame(dates × tickers)}  # factor scores
        returns_df:   DataFrame(dates × tickers)  # 1-day forward returns
        weights_dict: optional dict of factor weights
        """      
        self.file_dir = file_dir# = work_dir
        self.T = T
        self.T_momentum = T_momentum        
        self.index_name = index_name
        self.ticker_list = ticker_list
                       
        self.rets = returns_df
        self.ff_carhart_factors = ff_carhart_factors 
        self.factors = factors        
        self.factor_ics = factor_ics
        
        self.max_weight = max_weight
        self.cost_bps = cost_bps

        # Run everything automatically
        self.shift = 1
        self.score = self._combine_factors()        
        self.score.to_csv(file_dir+f'{index_name}_ticker_score_by_indicator.csv')
        
        self.weights_ls_mn = self._scores_to_weights(True) # long-short market neutral       
        self.weights_ls = self._scores_to_weights(False) # long-short but not market neutral
        self.weights_l = self._scores_to_weights_long() # long only        
        self.weights_ls_mn.to_csv(file_dir+f'{index_name}_ticker_weight_ls_mn_by_indicator_port.csv')
        self.weights_ls.to_csv(file_dir+f'{index_name}_ticker_weight_ls_by_indicator_port_.csv')
        self.weights_l.to_csv(file_dir+f'{index_name}_ticker_weight_l_by_indicator_port.csv')

        # long_short market neutral
        self.pnl_ls_mn, self.sharpe_ls_mn = self._backtest(self.weights_ls_mn, "long_short_mn")        
        # long-short but not market neurtal
        self.pnl_ls, self.sharpe_ls = self._backtest(self.weights_ls, "long_short")
        # long only
        self.pnl_l, self.sharpe_l = self._backtest(self.weights_l, "long_only")
        # portfolio pnl
        self.pnl_df = pd.concat([self.pnl_ls_mn, self.pnl_ls, self.pnl_l], axis=1)
        self.pnl_df.to_csv(file_dir+f'{index_name}_ticker_return_by_indicator_port.csv')

   # ---------------------------------------------------------
    def _align(self, dfs):
        """Align all DataFrames on common dates and tickers."""
        dfs = [d for d in dfs if d is not None]
        idx = dfs[0].index
        cols = dfs[0].columns
        for d in dfs[1:]:
            idx = idx.intersection(d.index)
            cols = cols.intersection(d.columns)
        return [d.reindex(index=idx, columns=cols) for d in dfs]

    # ---------------------------------------------------------
    def _combine_factors(self):
        """Combine factors linearly."""
        factors = self.factors
        ics = self.factor_ics.copy(deep=True)
        T_momentum = self.T_momentum
        shift = self.shift
        # factors = asset_alpha_portfolio_data.factors
        # ics = asset_alpha_portfolio_data.factor_ics        
        ics = ics.fillna(0.0)
        # weights[weights<0] = 0.0

        span_mean = 2*T_momentum
        win_var = 2*T_momentum
        var_floor = 1e-6

        # need to forecast weights 1 period foreward
        # 1. Expected IC (EMA)
        mean_ic = ics.ewm(span=span_mean, adjust=False, min_periods=T_momentum).mean()        
        # 2. Rolling variance of IC
        var_ic = ics.rolling(window=win_var, min_periods=T_momentum).var()       
        # 3. Compute risk-adjusted IC ratio per day
        w_raw = mean_ic / var_ic.clip(lower=var_floor)       
        # 4. Normalize across factors so that sum(|w|)=1 each day
        abs_sum = w_raw.abs().sum(axis=1).replace(0, np.nan)
        w_norm = w_raw.div(abs_sum, axis=0)        
        # 5. Shift forward by 1 day (forecast for t+1)
        w_forecast = w_norm.shift(shift)#.fillna(0.0)
        next_day = w_norm.index[-1] + Timedelta(days=1)
        # append a new last row (t+1) with today's weights
        last_row = w_norm.tail(1)
        last_row.index = [next_day]
        w_forecast = pd.concat([w_forecast, last_row], axis=0)
        w_forecast = w_forecast.fillna(0.0)

        # initialize result with zeros, same shape as any factor dataframe
        first_key = next(iter(factors))
        result = pd.DataFrame(0.0, index=factors[first_key].index, columns=factors[first_key].columns)
        
        # loop through all indicators (factors)
        for ind, df in factors.items():
            df = df.fillna(0)
            if ind in w_forecast.columns:  # ensure indicator has a weight
                # multiply each day's factor values by that day's weight
                w = w_forecast[ind]
                result += df.mul(w, axis=0)   # axis=0 means match by index (date)
    
        return result
    
    def _combine_factors_ori(self):
        """Combine factors linearly."""
        factors = self.factors
        weights_dict = self.weights_dict
        
        names = list(factors.keys())
        weights = weights_dict or {nm: 1.0 / len(names) for nm in names}

        idx, cols = None, None
        for f in factors.values():
            idx = f.index if idx is None else idx.intersection(f.index)
            cols = f.columns if cols is None else cols.intersection(f.columns)

        combo = pd.DataFrame(0.0, index=idx, columns=cols)
        for nm, df in factors.items():
            w = weights.get(nm, 0.0)
            combo = combo.add(df.reindex(index=idx, columns=cols) * w, fill_value=0.0)
        return combo    

    # ---------------------------------------------------------
    def _scores_to_weights(self, mn):
        """Turn daily factor scores into portfolio weights (long-short)."""
        score = self.score
        max_weight = self.max_weight
        # score = asset_alpha_portfolio_data.score
        # max_weight = asset_alpha_portfolio_data.max_weight
        
        w = score.divide(score.abs().sum(axis=1), axis=0)
        w = w.clip(-max_weight, max_weight)
        if mn: w = w.sub(w.mean(axis=1), axis=0) # market neutral
        gross = w.abs().sum(axis=1).replace(0, np.nan)        
        weights = w.divide(gross, axis=0)
        
        return weights
    
    def _scores_to_weights_long(self):
        # long only and not market neutral
        score = self.score
        max_weight = self.max_weight
        rf_col = self.rets.columns[-1]
        # score = asset_alpha_portfolio_data.score
        # max_weight = asset_alpha_portfolio_data.max_weight
        # rf_col = asset_alpha_portfolio_data.rets.columns[-1]
               
        # long only portfolio
        score_long = score.copy(deep=True)
        score_long[score_long<0] = 0
        
        w = score_long.divide(score_long.abs().sum(axis=1), axis=0)
        w = w.clip(-max_weight, max_weight)
        w[rf_col] = 1 - w.sum(axis=1)
        
        return w

    # ---------------------------------------------------------
    def _backtest(self, weights, port):
        """Simple backtest using 1-day forward returns."""
        shift = self.shift
        cost_bps = self.cost_bps
        # weights = self.weights
        rets = self.rets
        
        [weights, rets] = self._align([weights, rets])
        pnl = (weights.shift(shift) * rets).sum(axis=1)
        # pnl = (weights * rets).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1)
        costs = turnover * (cost_bps / 10000.0)
        pnl_net = pnl - costs
        ann_sharpe = pnl_net.mean() / pnl_net.std(ddof=1) * np.sqrt(self.T)
        pnl_net = pd.DataFrame(pnl_net, columns=[f'alpha_indicator_{port}_{self.index_name}'])
        
        return pnl_net, ann_sharpe
#%%