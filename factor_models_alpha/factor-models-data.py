#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECON367B Fall 2025 Multifactor Model Excercise
@author: raiyat
"""

import pandas as pd
import numpy as np
import yfinance as yf
import re
import statsmodels.api as sm
from SIM_and_CAPM_class import *

class asset_factor_data_class:
    def __init__(self, file_dir, T_momentum, index_name, ticker_list, asset_return_df):
        self.file_dir = file_dir
        self.T_momentum = T_momentum        
        self.index_name = index_name
        self.ticker_list = ticker_list
        self.asset_return_df = asset_return_df        
        # asset_return_df = asset_return_data_list[index_name].asset_return_df       
        # Filter dict: keep only items where value is in df.columns
        reduced_ticker_list = {k: v for k, v in ticker_list.items() if v in asset_return_df.columns}
        
        self.asset_fundamental_data_dict = {}
        for ticker in reduced_ticker_list.keys():
            print(ticker)        
            asset_fundamental_data = asset_fundamental_data_class(ticker, start="2010-01-01", filing_lag_days=45, verbose=False).build()
            asset_fundamental_data = asset_fundamental_data.ffill()
            asset_fundamental_data['book_to_market'] = asset_fundamental_data['book_value']/asset_fundamental_data['market_cap']
            asset_fundamental_data['return'] = asset_return_df[ticker_list[ticker]]
            asset_fundamental_data['return_momentum'] = (1 + asset_return_df[ticker_list[ticker]]).rolling(window=T_momentum).apply(np.prod, raw=True).shift(1) - 1
            self.asset_fundamental_data_dict[ticker] = asset_fundamental_data
            self.asset_fundamental_data_dict[ticker].to_csv(file_dir+index_name+'_'+ticker+'_ff_carhart.csv')
        
        big = pd.concat(self.asset_fundamental_data_dict, axis=1)   # panel is {ticker: df}
        self.factor_fundamental_data_dict = {f: big.xs(f, axis=1, level=1) for f in big.columns.levels[1]}
        for factor in self.factor_fundamental_data_dict.keys():
            self.factor_fundamental_data_dict[factor] = self.factor_fundamental_data_dict[factor].dropna(how='any')
            self.factor_fundamental_data_dict[factor].to_csv(file_dir+index_name+'_'+factor+'_ff_carhart.csv')
        
        if index_name == 'djia': m, n = 2, 3
        elif index_name == 'sp500': m, n = 5, 5
        elif index_name == 'sp400': m, n = 4, 5
        elif index_name == 'sp600': m, n = 6, 5
        else: m, n = 2, 3
            
        self.ff_carhart_factors = self.get_ff_carhart_factors(m, n)
        self.ff_carhart_factors.to_csv(file_dir+index_name+'_smb_hml_umb_factors.csv')
        
    def get_ff_carhart_factors(self, m, n):
        factor_fundamental_data_dict = self.factor_fundamental_data_dict
        # factor_fundamental_data_dict = asset_fundamental_data_list.factor_fundamental_data_dict
        
        # --- pull panels ---
        mcap = factor_fundamental_data_dict["market_cap"]
        btm  = factor_fundamental_data_dict["book_to_market"]
        mom  = factor_fundamental_data_dict["return_momentum"]
        ret  = factor_fundamental_data_dict["return"]
        
        # convert to DatetimeIndex, normalize, then to .date
        to_dt = lambda idx: pd.to_datetime(idx).tz_localize(None).normalize()
        
        dates = to_dt(mcap.index) \
            .intersection(to_dt(btm.index)) \
            .intersection(to_dt(mom.index)) \
            .intersection(to_dt(ret.index))
        
        # convert to pure date objects
        dates = pd.Index(dates.date)
        
        # tickers
        tickers = mcap.columns.intersection(btm.columns).intersection(mom.columns).intersection(ret.columns)
        
        # realign all four DataFrames
        mcap = mcap.loc[dates, tickers]
        btm  = btm.loc[dates, tickers]
        mom  = mom.loc[dates, tickers]
        ret  = ret.loc[dates, tickers]
    
        # # --- grid sizes ---
        # m, n = 2, 3   # e.g. size split=2, style split=3
        
        # --- quantile labels (1..q per row) ---
        def qlabels(df, q):
            return df.apply(
                lambda r: pd.qcut(r.dropna(), q, labels=range(1, q+1), duplicates="drop")
                             .reindex(df.columns), axis=1
            )
        
        size_q = qlabels(mcap, m)
        bm_q   = qlabels(btm, n)
        mom_q  = qlabels(mom, n)
        
        # --- define masks ---
        if m % 2 == 0:  # SMB split
            small_mask, big_mask = size_q.le(m//2), size_q.gt(m//2)
        else:  # drop middle row if odd split
            k = m//2
            small_mask, big_mask = size_q.le(k), size_q.ge(k+2)
        
        high_bm, low_bm = bm_q.eq(n), bm_q.eq(1)   # HML
        up_mom, dn_mom  = mom_q.eq(n), mom_q.eq(1) # UMD
        
        # --- helper: value-weighted average ---
        def wavg(r, w):
            r, w = r.dropna(), w.loc[r.index]
            if len(r) == 0 or w.sum() == 0: 
                return np.nan
            return np.average(r, weights=w)
        
        # --- use lagged market cap as weights ---
        mcap_lag = mcap.shift(1)
        
        SMB, HML, UMD = [], [], []
        
        for d in dates:
            r, w = ret.loc[d], mcap_lag.loc[d]
        
            smb_s = wavg(r[small_mask.loc[d]], w[small_mask.loc[d]])
            smb_b = wavg(r[big_mask.loc[d]], w[big_mask.loc[d]])
            hml_h = wavg(r[high_bm.loc[d]], w[high_bm.loc[d]])
            hml_l = wavg(r[low_bm.loc[d]], w[low_bm.loc[d]])
            umd_u = wavg(r[up_mom.loc[d]], w[up_mom.loc[d]])
            umd_d = wavg(r[dn_mom.loc[d]], w[dn_mom.loc[d]])
        
            SMB.append(smb_s - smb_b)
            HML.append(hml_h - hml_l)
            UMD.append(umd_u - umd_d)
        
        factors = pd.DataFrame({"SMB": SMB, "HML": HML, "UMD": UMD}, index=dates)
        factors = factors.dropna(how='all')
        
        return factors
        
class asset_fundamental_data_class:
    """
    Build daily series for a stock (clean + easy version):
      - price ('Close', split-adjusted by Yahoo)
      - shares (backfilled from latest snapshot using split history)
      - market_cap (price * shares)
      - book_value (annual + quarterly from Yahoo, merged; forward-filled to daily; optional filing lag)
      - book_to_market (book_value / market_cap)

    Output index is plain DATE objects (YYYY-MM-DD), not datetimes.
    Leading rows with ANY NaN across columns are removed.
    """

    # Regex patterns to find the right balance-sheet rows regardless of wording
    BOOK_PATTERNS  = [
        r"total\s+stockholder[s]?\s+equity",
        r"stockholder[s]?\s+equity",
        r"total\s+shareholder[s]?\s+equity",
        r"shareholder[s]?\s+equity",
        r"total\s+equity",
        r"common\s+stock\s+equity",
        r"total\s+equity\s+gross\s+minority\s+interest",
        r"equity\s+attributable\s+to\s+parent",
    ]
    ASSET_PATTERNS = [
        r"total\s+assets",
        r"total\s+assets\s+net\s+minority\s+interest",
    ]
    LIAB_PATTERNS  = [
        r"total\s+liabilities\s+net\s+minority\s+interest",
        r"total\s+liabilities",
    ]

    def __init__(self, ticker, start="2000-01-01", end=None, filing_lag_days=0, verbose=False):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.filing_lag_days = int(filing_lag_days) if filing_lag_days is not None else 0
        self.verbose = verbose

    # ---------- utilities ----------
    @staticmethod
    def _naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Force a timezone-naive DatetimeIndex."""
        if getattr(idx, "tz", None) is not None:
            try:
                return idx.tz_convert("UTC").tz_localize(None)
            except Exception:
                return idx.tz_localize(None)
        return idx

    # ---------- public ----------
    def build(self) -> pd.DataFrame:
        T = yf.Ticker(self.ticker)

        # 1) Prices (use 'Close'); normalize to daily dates and tz-naive
        px = T.history(start=self.start, end=self.end, actions=True)
        if px.empty:
            raise ValueError("No price data from Yahoo.")
        if "Close" not in px.columns:
            raise KeyError("Yahoo data has no 'Close' column.")

        px.index = self._naive(px.index).normalize()
        price = px["Close"].rename("price")
        splits = px["Stock Splits"] if "Stock Splits" in px.columns else pd.Series(0.0, index=price.index)

        # 2) Shares (latest snapshot; fallback from marketCap/last price), back-adjust via splits
        info = T.info or {}
        latest_shares = info.get("sharesOutstanding", np.nan)
        if (pd.isna(latest_shares) or latest_shares <= 0) and info.get("marketCap", 0) and price.iloc[-1] > 0:
            latest_shares = float(info["marketCap"]) / float(price.iloc[-1])
        shares = self._daily_shares_from_latest(price.index, latest_shares, splits).rename("shares")

        # 3) Market cap
        mcap = (price * shares).rename("market_cap")

        # 4) Book value points (annual + quarterly) -> daily (tz-naive align, filing lag optional)
        be_points = self._book_points_yahoo(T)  # Series at report dates (naive)
        book = self._book_to_daily(be_points, price.index, self.filing_lag_days).rename("book_value")

        # 5) Book-to-Market
        btm = (book / mcap).rename("book_to_market")

        # Combine
        df = pd.concat([price, shares, mcap, book, btm], axis=1)

        # Clean: drop rows that are all NaN; then drop leading rows with ANY NaN in any column
        df = df.dropna(how="all")
        full_row_mask = ~df.isna().any(axis=1)  # rows where all columns are present
        if full_row_mask.any():
            first_full_idx = full_row_mask.idxmax()  # first date with no NaNs in any column
            df = df.loc[first_full_idx:]
        else:
            df = df.iloc[0:0]

        # >>> Make index DATE-only (no time): e.g., 2021-01-01
        df.index = df.index.date

        return df

    # ---------- helpers ----------
    def _match_row_ci(self, df: pd.DataFrame, patterns):
        """Find first row whose name matches any regex (case-insensitive). Return that row as Series."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        idx = [str(x) for x in df.index]
        for pat in patterns:
            rgx = re.compile(pat, flags=re.IGNORECASE)
            for i, name in enumerate(idx):
                if rgx.search(name):
                    return df.iloc[i]
        return None

    def _assets_minus_liabs(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        a = self._match_row_ci(df, self.ASSET_PATTERNS)
        l = self._match_row_ci(df, self.LIAB_PATTERNS)
        if a is None or l is None:
            return None
        try:
            return (a.astype("float64") - l.astype("float64"))
        except Exception:
            return None

    def _row_to_series(self, row: pd.Series):
        if row is None:
            return None
        s = row.copy()
        s.index = pd.to_datetime(s.index)
        s.index = self._naive(s.index)
        s = s.sort_index()
        return s.astype("float64")

    def _book_points_yahoo(self, T) -> pd.Series:
        """Merge annual + quarterly book equity; fallback to Assets - Liabilities; return longest Series."""
        try:
            bs_q = T.quarterly_balance_sheet
        except Exception:
            bs_q = pd.DataFrame()
        try:
            bs_a = T.balance_sheet
        except Exception:
            bs_a = pd.DataFrame()

        if self.verbose:
            if isinstance(bs_q, pd.DataFrame):
                print("Quarterly rows:", list(bs_q.index))
            if isinstance(bs_a, pd.DataFrame):
                print("Annual rows:", list(bs_a.index))

        be_q = self._row_to_series(self._match_row_ci(bs_q, self.BOOK_PATTERNS))
        if be_q is None:
            be_q = self._row_to_series(self._assets_minus_liabs(bs_q))

        be_a = self._row_to_series(self._match_row_ci(bs_a, self.BOOK_PATTERNS))
        if be_a is None:
            be_a = self._row_to_series(self._assets_minus_liabs(bs_a))

        parts = []
        if be_a is not None and not be_a.empty:
            parts.append(be_a.rename("book_equity"))
        if be_q is not None and not be_q.empty:
            parts.append(be_q.rename("book_equity"))

        if not parts:
            if self.verbose:
                print("No book equity found for", self.ticker)
            return pd.Series(dtype="float64", name="book_equity")

        be = pd.concat(parts).sort_index()
        be = be[~be.index.duplicated(keep="last")]  # prefer the later-added series (usually quarterly)
        be.name = "book_equity"
        return be

    def _book_to_daily(self, be_series: pd.Series, daily_index: pd.DatetimeIndex, lag_days: int) -> pd.Series:
        """Forward-fill book equity onto the daily calendar; apply filing lag if requested."""
        daily_index = self._naive(pd.DatetimeIndex(daily_index)).normalize()

        if be_series is None or be_series.empty:
            return pd.Series(index=daily_index, dtype="float64", name="book_value")

        be = be_series.copy()
        if lag_days and lag_days > 0:
            be.index = be.index + pd.to_timedelta(lag_days, unit="D")

        be.index = self._naive(pd.DatetimeIndex(be.index)).normalize()
        out = be.reindex(daily_index, method="ffill")
        out.name = "book_value"
        return out

    def _daily_shares_from_latest(self, index: pd.DatetimeIndex, latest_shares: float, splits: pd.Series) -> pd.Series:
        """Backfill shares from the latest snapshot using split ratios (no issuance/buyback modeling)."""
        index = self._naive(pd.DatetimeIndex(index)).normalize()
        if isinstance(splits, pd.Series):
            splits.index = self._naive(pd.DatetimeIndex(splits.index)).normalize()

        s = pd.Series(index=index, dtype="float64")
        if not latest_shares or pd.isna(latest_shares) or latest_shares <= 0:
            s[:] = np.nan
            return s

        s.iloc[-1] = float(latest_shares)
        sp = splits.reindex(index).fillna(0.0)

        # If split ratio r is stamped on day t, shares(t-1) = shares(t) / r
        for i in range(len(index) - 2, -1, -1):
            r = sp.iloc[i + 1]
            if r and r > 0:
                s.iloc[i] = s.iloc[i + 1] / r
            else:
                s.iloc[i] = s.iloc[i + 1]
        return s

class asset_multi_factor_class:
    def __init__(self, file_dir, T, index_name, 
                 asset_return_data, asset_factor_data):
        self.file_dir = file_dir
        self.T = T
        self.p_lim = 0.05
        self.shift = 1
        self.index_name = index_name
        self.asset_return_class = asset_return_data        
        self.return_df = asset_return_data.asset_return_df
        self.return_rf = asset_return_data.asset_return_df_rf
        self.return_index = asset_return_data.asset_return_index_df
        
        self.asset_factor_data = asset_factor_data
        self.return_factors = asset_factor_data.ff_carhart_factors
        
        self.capm_ticker_dates_dict, self.capm_return_dates_dict = self.multi_factor_model_regressions('capm')
        self.ff3_ticker_dates_dict, self.ff3_return_dates_dict = self.multi_factor_model_regressions('ff3')
        self.carhart_ticker_dates_dict, self.carhart_return_dates_dict = self.multi_factor_model_regressions('carhart')
    
        # capm
        self.capm_weight_0_dates, self.capm_weight_adj_index_dates = self.multi_factor_model_weight(self.capm_return_dates_dict, model_choice='capm')              
        self.capm_return_portfolio, self.capm_return_portfolio_active_index = self.multi_factor_portfolio_performance(self.capm_weight_adj_index_dates, model_choice='camp')
    
        # fama-french 3-factor
        self.ff3_weight_0_dates, self.ff3_weight_adj_index_dates = self.multi_factor_model_weight(self.ff3_return_dates_dict, model_choice='ff3')              
        self.ff3_return_portfolio, self.ff3_return_portfolio_active_index = self.multi_factor_portfolio_performance(self.ff3_weight_adj_index_dates, model_choice='ff3')

        # carhart 4-factor
        self.carhart_weight_0_dates, self.carhart_weight_adj_index_dates = self.multi_factor_model_weight(self.carhart_return_dates_dict, model_choice='carhart')              
        self.carhart_return_portfolio, self.carhart_return_portfolio_active_index = self.multi_factor_portfolio_performance(self.carhart_weight_adj_index_dates, model_choice='carhart')

        # performance evaluation and comparisons of 3 portfolios
        self.portfolio_performance_df = pd.concat([self.capm_return_portfolio_active_index,
                                               self.ff3_return_portfolio_active_index,
                                               self.carhart_return_portfolio_active_index],
                                               axis=1)
        self.portfolio_performance_df.columns = pd.Index([f'capm_{index_name}', f'ff3_{index_name}', f'carhart_{index_name}'])
        self.portfolio_performance_df.to_csv(file_dir+index_name+'_factor_model_portfolio_performance.csv')

        self.portfolio_performance = multi_factor_portfolio_performance_class(file_dir, T, index_name, 'ff3_carhart',
                                                                              self.portfolio_performance_df, self.return_index, self.return_rf)

    def multi_factor_model_regressions(self, model_choice):
        file_dir = self.file_dir
        T = self.T
        index_name = self.index_name
        p_lim = self.p_lim
        return_df = self.return_df # return data for all qualified stocks in index
        return_index = self.return_index # return data for the relevant index
        return_rf = self.return_rf # return data for risk fee asset
        return_factors = self.return_factors
        
        lookback = T            
        freq = T #if cov_by_date else 1.0                                                                  
                
        return_df = return_df.loc[return_factors.index]
        return_index = return_index.loc[return_factors.index]
        return_rf = return_rf.loc[return_factors.index]
        
        # set up the linear regressions
        return_df_ex = return_df - return_rf.values
        return_index_ex = return_index - return_rf.values
        return_index_factors = pd.concat([return_index_ex, return_factors], axis=1)
        
        # rolling window is of T, 252 days
        dates = return_df.index[T-1:]
    
        ticker_dates_dict = {}
        # loop through the assets
        ## first, just capm
        if model_choice == 'capm': # single factor capm
            X = return_index_factors[index_name]
            X = X.to_frame()
        elif model_choice == 'ff3': # fama-french 3-factor
            X = return_index_factors[[index_name, 'SMB', 'HML']]
        elif model_choice == 'carhart': # carhart 4-factor
            X = return_index_factors      
            
        for return_ticker in return_df_ex.columns:
            # return_ticker = return_df_ex.columns[0]
            Y = return_df_ex[return_ticker]
            # X = return_index_ex
            
            # each date in the dates
            ticker_dt_list = []
            for dt in dates:
                # dt = dates[0]
                # print(f"on date {dt}")
                # find the integer position of dt in Y's index
                end_pos = Y.index.get_loc(dt)               
                # slice from T rows before up to dt
                y = Y.iloc[end_pos-lookback+1: end_pos+1]
                x = sm.add_constant(X.iloc[end_pos-lookback+1: end_pos+1])
                model = sm.OLS(y, x).fit()
                
                # pull all factor stats by label (skip 'const')
                factors = list(X.columns)                
                betas   = model.params[factors]
                betas_se= model.bse[factors]
                betas_t = model.tvalues[factors]
                betas_p = model.pvalues[factors]
                
                # residual               
                # collect results in dict
                dt_results = {
                    "alpha": model.params["const"],
                    "alpha_p": model.pvalues["const"],
                    "alpha_se": model.bse["const"],                    
                    "alpha_t": model.tvalues["const"],      
                    "alpha_sig": model.params["const"]if model.pvalues["const"]<=p_lim else 0.0,

                    # "beta": model.params[1],
                    # "beta_p": model.pvalues[1],
                    # "beta_se": model.bse[1],
                    # "beta_t": model.tvalues[1],
                    
                    "R2": model.rsquared,
                    "Adj_R2": model.rsquared_adj,
                    
                    "ResVar": np.var(model.resid, ddof=1), # unbiased variance of residuals
                    "ResStd": np.std(model.resid, ddof=1), # unbiased std of residuals
                    
                    "DW": sm.stats.stattools.durbin_watson(model.resid),
                    "Nobs": int(model.nobs),
                    "Fstat": model.fvalue,
                    "F_p": model.f_pvalue,
                    
                    # 'u_m': (X.iloc[end_pos-lookback+1: end_pos+1]).mean().iloc[0],
                    # 'var_m': (X.iloc[end_pos-lookback+1: end_pos+1]).var().iloc[0]
                }
                
                # flatten factor stats into the dict, e.g. beta_MKT, beta_SMB, ...
                for f in factors:
                    dt_results[f"beta_{f}"]   = float(betas[f])
                    dt_results[f"beta_{f}_se"]= float(betas_se[f])
                    dt_results[f"beta_{f}_t"] = float(betas_t[f])
                    dt_results[f"beta_{f}_p"] = float(betas_p[f])
                    dt_results[f'beta_{f}_sig'] = float(betas[f]) if float(betas_p[f])<=p_lim else 0.0
                
                # 1×n DataFrame
                dt_results = pd.DataFrame([dt_results], index=[dt])
                # dt_results['alpha_sig'] = dt_results['alpha'].values[0] if dt_results['alpha_p'].values[0]<=p_lim else 0.0                
                # dt_results['beta_sig'] = dt_results['beta'].values[0] if dt_results['beta_p'].values[0]<=p_lim else 0.0
                # dt_results['u/var_m'] = dt_results['u_m'] / dt_results['var_m']

                ticker_dt_list.append(dt_results)
                
            ticker_dt_list = pd.concat(ticker_dt_list, axis=0)
            ticker_dt_list.name = return_ticker
            ticker_dates_dict[return_ticker] = ticker_dt_list
            ticker_dt_list.to_csv(file_dir+index_name+'_'+model_choice+'_'+return_ticker.replace('/','_')+'_ticker_mfm.csv')
                          
        panel = pd.concat(ticker_dates_dict, axis=1)  # MultiIndex columns: (ticker, old_col)
        fields = panel.columns.get_level_values(1).unique()       
        # return_dates_dict = {f: panel.xs(f, axis=1, level=1) for f in fields}        
        return_dates_dict = {}
        for f in fields:
            return_dates_dict[f] = panel.xs(f, axis=1, level=1)
            f_name = f.replace("/", "_")
            return_dates_dict[f].to_csv(file_dir+index_name+'_'+model_choice+'_'+f_name+'_parameter_mfm.csv')
        
        return ticker_dates_dict, return_dates_dict

    def multi_factor_model_weight(self, return_dates_dict, model_choice):        
        return_df = self.return_df
        file_dir = self.file_dir
        index_name = self.index_name
        weight_lim = 1.0
        
        # alpha
        alpha_0_dates = return_dates_dict['alpha_sig'].copy(deep=True)
        alpha_0_dates[alpha_0_dates<0] = 0   
        
        # w0 weight - initial positions
        var_0_dates = return_dates_dict['ResVar'].copy(deep=True)
        weight_0_dates = alpha_0_dates/var_0_dates              
        weight_0_dates['total'] = weight_0_dates.sum(axis=1)        
        
        mask = weight_0_dates["total"] != 0
        weight_0_dates.loc[mask, weight_0_dates.columns] = weight_0_dates.loc[mask, weight_0_dates.columns].div(weight_0_dates.loc[mask, "total"], axis=0)
        weight_0_dates[index_name] = 1.0 - weight_0_dates['total']
        
        weight_adj_index_dates = weight_0_dates.drop(columns='total')
        weight_adj_index_dates.to_csv(file_dir+index_name+'_'+model_choice+'_weight_adj_with_index.csv')

        return weight_0_dates, weight_adj_index_dates

    def multi_factor_portfolio_performance(self, weight_adj_index_dates, model_choice):
        file_dir = self.file_dir
        T = self.T
        shift = self.shift
        index_name = self.index_name
        p_lim = self.p_lim
        return_df = self.return_df # return data for all qualified stocks in index
        return_index = self.return_index # return data for the relevant index
        return_rf = self.return_rf # return data for risk fee asset
        weight_adj_index_dates = weight_adj_index_dates.copy(deep=True)
        
        return_df = return_df.loc[weight_adj_index_dates.index]
        return_index = return_index.loc[weight_adj_index_dates.index]      
        return_df_index = pd.concat([return_df, return_index], axis=1)
        
        return_portfolio = (weight_adj_index_dates.shift(shift) * return_df_index).dropna(how='all')
        return_portfolio['total'] = return_portfolio.sum(axis=1) 
        return_portfolio_active_index = return_portfolio['total'].to_frame()
        return_portfolio_active_index.columns = pd.Index([f'active_{index_name}'])
        
        return_portfolio.to_csv(file_dir+index_name+'_'+model_choice+'_portfolio_return_all.csv')
        return_portfolio_active_index.to_csv(file_dir+index_name+'_'+model_choice+'_'+'portfolio_return_active_index.csv')
                                                                         
        return return_portfolio, return_portfolio_active_index

class multi_factor_portfolio_performance_class:
    def __init__(self, file_dir, T, index_name, model_name, portfolio_performance_df,return_index_df,return_rf):
        self.start = 100.0 # initial portfolio wealth
        self.p_lim = 0.05
        self.file_dir = file_dir
        self.T = T
        self.index_name = index_name
        self.model_name = model_name
        self.portfolio_performance_df = portfolio_performance_df
        self.return_index_df = return_index_df
        self.return_rf = return_rf
        
        self.return_index_df, self.return_rf = self.line_up_data()
        # calculating performance metrics
        self.returns_df, self.prices_df = self.compute_prices()
        
        self.sharpe_ratio_df = self.compute_sharpe()
        self.sortino_ratio_df = self.compute_sortino()
        self.drawdowns_df = self.compute_drawdowns()
        self.max_drawdown_df = self.compute_max_drawdown()
        
        self.sim_regression_data = self.sim_return_data()       
        self.return_df_annualized, self.treynor_ratio = self.compute_treynor()        
        
        self.alpha_list = [0.95, 0.99] # VaR confidence level
        self.var_target_list = [0.05, 0.1] # VaR target
        self.portfolio_stats = self.returns_df.apply(self.asset_return_des_stats, axis=0)
        
        self._prices_plot = self.plot_prices()
        self._returns_plot = self.plot_returns()
        self._drawdown_plot = self.plot_drawdowns()
        
        self.performance_table = self.metrics_table(annualize=True)
                
    # line up the data   
    def line_up_data(self):
        portfolio_performance_df = self.portfolio_performance_df
        return_index_df = self.return_index_df
        return_rf = self.return_rf
        
        return_index_df = return_index_df.loc[portfolio_performance_df.index]
        return_rf = return_rf.loc[portfolio_performance_df.index]
        
        return return_index_df,return_rf

    def compute_prices(self):
        file_dir = self.file_dir
        index_name = self.index_name
        model_name = self.model_name
        portfolio_performance_df = self.portfolio_performance_df
        return_index_df = self.return_index_df
        start = self.start
        
        returns_df = pd.concat([portfolio_performance_df, return_index_df], axis=1)
        
        # Start all price series at 100        
        prices_df = (1 + returns_df).cumprod() * start
        prices_df.to_csv(file_dir+index_name+f'_{model_name}_portfolio_price.csv', index=True, index_label='date')
        return returns_df, prices_df

    def compute_sharpe(self):
        returns_df = self.returns_df
        rf_df = self.return_rf
        excess = returns_df.sub(rf_df.squeeze(), axis=0)
        sharpe_ratios = excess.mean() / excess.std(ddof=1)
        return sharpe_ratios

    def compute_sortino(self):
        returns_df = self.returns_df
        rf_df = self.return_rf
        excess = returns_df.sub(rf_df.squeeze(), axis=0)
        downside = excess[excess < 0]
        sortino_ratios = excess.mean() / downside.std(ddof=1)
        return sortino_ratios
    
    def compute_drawdowns(self):
        file_dir = self.file_dir
        index_name = self.index_name
        model_name = self.model_name
        prices = self.prices_df
        rolling_max = prices.cummax()
        drawdowns = prices / rolling_max - 1
        drawdowns.to_csv(file_dir+index_name+f'_{model_name}_portfolio_drawdown.csv', index=True, index_label='date')
        return drawdowns

    def compute_max_drawdown(self):
        prices = self.prices_df
        roll_max = prices.cummax()
        drawdowns = (prices - roll_max) / roll_max
        max_drawdowns = drawdowns.min()
        return max_drawdowns
        
    def sim_return_data(self):
        return_df = self.portfolio_performance_df
        return_index = self.return_index_df
        return_rf = self.return_rf
        p_lim = self.p_lim
        
        # set up the linear regressions
        A = return_df_ex = return_df - return_rf.values
        B = return_index_ex = return_index - return_rf.values
        
        """
        For each column a in A, run OLS: A[a] ~ const + B[b],
        where b is the (single) column in B whose name is a substring of a.
        Assumes clean data (exactly one match per A column).
        Returns a metrics-by-columns DataFrame.
        """
        # Map A->B via substring match
        mapping = {a: next(b for b in B.columns if b in a) for a in A.columns}
    
        def fit_one(colA: str):
            colB = mapping[colA]
            y = A[colA]
            x = B[colB]
            m = y.notna() & x.notna()
    
            Xmat = pd.DataFrame({"const": 1.0, "x": x[m].to_numpy()}, index=x[m].index)
            model = sm.OLS(y[m].to_numpy(), Xmat).fit()
    
            return colA, {
                "alpha":   model.params["const"],
                "alpha_p": model.pvalues["const"],
                "alpha_se":model.bse["const"],
                "alpha_t": model.tvalues["const"],
    
                "beta":    model.params["x"],
                "beta_p":  model.pvalues["x"],
                "beta_se": model.bse["x"],
                "beta_t":  model.tvalues["x"],
    
                "R2":      model.rsquared,
                "Adj_R2":  model.rsquared_adj,
    
                "ResVar":  np.var(model.resid, ddof=1),
                "ResStd":  np.std(model.resid, ddof=1),
    
                "DW":      sm.stats.stattools.durbin_watson(model.resid),
                "Nobs":    int(model.nobs),
                "Fstat":   model.fvalue,
                "F_p":     model.f_pvalue,
            }
    
        with ThreadPoolExecutor(max_workers=min(len(mapping), 8)) as ex:
            results = dict(ex.map(fit_one, mapping.keys()))
        results = pd.DataFrame(results).T
        results["alpha_sig"] = results["alpha"].where(results["alpha_p"]<=p_lim, 0.0)
        results["beta_sig"]  = results["beta"].where(results["beta_p"]<=p_lim, 0.0)
    
        return results.T
    
    def compute_treynor(self):
        """
        Treynor ratio = (E[R_p] - R_f) / beta_p
        market_returns: Series of market benchmark returns (aligned with portfolio dates)
        """
        return_df = self.portfolio_performance_df
        return_index = self.return_index_df
        return_rf = self.return_rf
        sim_regression_data = self.sim_regression_data
        T = self.T
        
        # return_df = portfolio_performance.portfolio_performance_df
        # return_index = portfolio_performance.return_index_df
        # return_rf = portfolio_performance.return_rf 
        # sim_regression_data = portfolio_performance.sim_regression_data
        
        # set up the linear regressions
        return_df_ex = return_df - return_rf.values
        return_index_ex = return_index - return_rf.values
            
        return_rf_total = (1 + return_rf).prod()
        return_rf_annualized = return_rf_total**(T/len(return_rf)) - 1

        return_df_total = (1 + return_df).prod()
        return_df_annualized = (return_df_total**(T/len(return_df)) - 1).to_frame().T
        return_df_annualized.index = pd.Index(['Annualized Return (current sample)'])
        
        treynor_ratio = return_df_annualized / sim_regression_data.loc['beta_sig']
        treynor_ratio.index = pd.Index(['Treynor annualized'])
        
        return return_df_annualized, treynor_ratio
        
    def metrics_table(self, annualize=False):        
        """
        Return a compact metrics table with rows = ['Sharpe','Sortino','MaxDrawdown']
        and columns = portfolio names (same as in self.returns_df).
        Set annualize=True to annualize Sharpe/Sortino using self.T.
        """
        file_dir = self.file_dir
        T = self.T
        index_name = self.index_name
        model_name = self.model_name
        returns_df = self.returns_df
        sharpe = self.sharpe_ratio_df.copy()
        sortino = self.sortino_ratio_df.copy()
        mdd = self.max_drawdown_df.copy()  # already negative numbers
        return_df_annualized = self.return_df_annualized
        treynor_ratio = self.treynor_ratio
        portfolio_stats = self.portfolio_stats
        sim_regression_data = self.sim_regression_data
    
        if annualize:
            # total sample annualized return
            # return_annualized = returns_df.mean(skipna=True) * T                       
            returns_total = (1 + returns_df).prod()
            return_annualized = (returns_total**(T/len(returns_df)) - 1)

            k = np.sqrt(T)
            vol_annualized = returns_df.std(skipna=True, ddof=1) * k            
            sharpe  = sharpe * k
            sortino = sortino * k
    
        tbl = pd.DataFrame({
            "Return annualized (total sample)": return_annualized,
            "Volatility annualized": vol_annualized,
            "Sharpe annualized": sharpe,
            "Sortino annualized": sortino,
            "Max Drawdown": mdd
        }).T

        # clean up any inf/-inf from zero stdev cases
        tbl = tbl.replace([np.inf, -np.inf], np.nan)
        tbl = pd.concat([tbl, return_df_annualized, treynor_ratio]).sort_index()
        tbl = pd.concat([tbl, sim_regression_data])
        
        tbl.to_csv(file_dir+index_name+f'_{model_name}_multi_factor_portfolio_performance.csv')
        
        return tbl

    def plot_prices(self):
        file_dir = self.file_dir
        prices_df = self.prices_df
        index_name = self.index_name
        model_name = self.model_name
        
        title = f"{index_name}_{model_name}_Portfolio and Index Prices"
        prices_df.plot(figsize=(10, 6), title=title)
        plt.ylabel("Price")
        plt.grid(True) 
        plt.savefig(file_dir+title, dpi=300, bbox_inches="tight")
        plt.show() 

        return True
    
    def plot_returns(self):
        file_dir = self.file_dir
        returns_df = self.portfolio_performance_df
        index_name = self.index_name
        model_name = self.model_name
        
        title = f"{index_name}_{model_name}_Portfolio and Index Returns"
        returns_df.plot(figsize=(10, 6), title=title)
        plt.ylabel("Return")
        plt.grid(True) 
        plt.savefig(file_dir+title, dpi=300, bbox_inches="tight")
        plt.show() 

        return True

    def plot_drawdowns(self):
        file_dir = self.file_dir
        drawdowns_df = self.drawdowns_df
        index_name = self.index_name
        model_name = self.model_name
        
        title = f"{index_name}_{model_name}_Portfolio and Index Drawdowns"
        drawdowns_df.plot(figsize=(10, 6), title=title)
        plt.ylabel("Drawdown")
        plt.grid(True)
        plt.savefig(file_dir+title, dpi=300, bbox_inches="tight")        
        plt.show() 
        
        return True
    
    # function to get the asset return descriptive statistics
    def asset_return_des_stats(self, ticker_df): 
        alpha_list = self.alpha_list
        var_target_list = self.var_target_list
        # ticker_df = portfolio_performance.returns_df
        # print(ticker_df.columns)
        
        ticker_skewness = ticker_df.skew()
        ticker_kurtosis = stats.kurtosis(ticker_df, fisher=True)
        ticker_describe = ticker_df.describe()
                
        # A) Given confidence, get VaR & CVaR
        ticker_var_list = []
        ticker_cvar_list = []
        for alpha in alpha_list:
            # alpha = alpha_list[0]
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
#%%