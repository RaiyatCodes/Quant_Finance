#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECON369 Spring 2026 factor/indicator, test and portfolio, fama-macbeth excercise
@author: raiyat
"""

import os
work_dir = "/Users/dengqi/Documents/Documents/AC - Gettysburg/2026-Spring/ECON369/Assignment_2/" 
os.chdir(work_dir)
os.getcwd()

import pandas as pd
import pandas_datareader as web # Datareader to download price data from Yahoo Finance
import statsmodels.api as smf # Statsmodels to run multivariate regressions
import urllib.request # To download the Fama French data from the web
import zipfile # To unzip the ZipFile 
from datetime import datetime

# import our own classes
from Portfolio_Optimization_class import *
from SIM_and_CAPM_class import *
from CAPM_FF_data_class import *
from factors_tests_class import *
from fama_macbeth_class import *

# ----------------------- factors/indicators model main -----------------------
if __name__ == "__main__":    
    # only need data after 2010-01-01
    T = 252 # number of trading days per year
    start_date = datetime(2010, 1, 1).date()
    end_date = datetime(2030, 12, 31).date()
    
    # large cap, mid cap, small cap, and dow jones
    index_list = ['djia']
    # index_list = ['sp500']
    # index_list = ['sp400']
    # index_list = ['sp600']
    # index_list = ['sp500', 'sp400', 'sp600', 'djia']
    asset_list_data = asset_list_data_class(work_dir, index_list)     

    # for each index, run the ticker and down data from yahoo     
    # if a stock listed after 2022-01-10, ignore it, not enough historical data
    latest_start_date = datetime(2022, 1, 1).date() # datetime(2020, 1, 1).date()
    ticker_index_list = {'^GSPC':'sp500',
                         '^SP400': 'sp400',
                         '^SP600': 'sp600', 
                         '^DJI': 'djia'                         
                         }    
    
    # trade on daily basis, look back 10 days for factor/indicator construction
    T_momentum = 10 
    asset_return_data_list = {}
    asset_factor_data_list = {}
    asset_multi_factor_data_list = {}
    asset_indicator_data_list = {}
    asset_alpha_portfolio_data_list = {}
    asset_alpha_portfolio_performance_data_list = {}
    asset_fama_macbeth_data_list = {}
    asset_fama_macbeth_portfolio_list = {}
    asset_alpha_fama_macbeth_portfolio_performance_data_list = {}
       
    for index_name in asset_list_data.stock_ranked_dict_list.keys():
        print (index_name)       
        ticker_list = asset_list_data.stock_ranked_dict_list[index_name]        
        # risk free and indices
        ticker_rf = 'SOFR'
        # index        
        ticker_index = {k:v for k, v in ticker_index_list.items() if v == index_name}
        print(ticker_index)
        
        # all asset data in class
        is_portfolio = False
        asset_return_data = asset_return_data_class(
            work_dir, index_name, T, is_portfolio,
            ticker_list, ticker_rf, ticker_index,
            start_date, end_date, latest_start_date
            )        
        asset_return_data_list[index_name] = asset_return_data   
 
    # for each index, build the daily series for market cap, book value, and momentum    
    ## ff-3 and carhart-4 factors: mkt, smb, hml, and umd       
        # get all asset factors
        asset_factor_data = asset_factor_data_class(
            work_dir, T_momentum, index_name, ticker_list,
            asset_return_data.asset_return_df
            )        
        asset_factor_data_list[index_name] = asset_factor_data

    # now let's do linear regression and portfolio selection
        asset_multi_factor_data = asset_multi_factor_class(
            work_dir, T, index_name, 
            asset_return_data,
            asset_factor_data)        
        asset_multi_factor_data_list[index_name] = asset_multi_factor_data

    # for each index, build the daily series for indicators and factors
    ## use talib for indicators, and use alphalens for portfoio construction
    ## and use fama-macbeth for risk premium
        asset_indicator_data = asset_indicator_data_class(
            work_dir, T, T_momentum, index_name, ticker_list,
            asset_return_data.asset_return_df,
            asset_return_data.asset_close_df,
            asset_return_data.asset_ticker_df_dict,
            asset_return_data.asset_index_df_dict,
            asset_factor_data.asset_fundamental_data_dict,
            asset_factor_data.ff_carhart_factors
            )
        asset_indicator_data_list[index_name] = asset_indicator_data

    # now, create the alphalens portfolio
    # 1D forecast
        asset_alpha_portfolio_data = asset_alpha_portfolio_data_class(
            work_dir, T, T_momentum, index_name, ticker_list,
            returns_df = asset_return_data.asset_return_combined, #df,
            ff_carhart_factors = asset_factor_data.ff_carhart_factors,
            factors = asset_indicator_data.parameter_all_indicator_dict,
            factor_ics = asset_indicator_data.horizon_ic_t_ts_dict['1D']
            )
        asset_alpha_portfolio_data_list[index_name] = asset_alpha_portfolio_data

    # now let's do linear regression and portfolio selection
        asset_alpha_portfolio_performance_data = multi_factor_portfolio_performance_class(
            work_dir, T, index_name, 'talib_indicator',
            asset_alpha_portfolio_data.pnl_df, 
            asset_return_data.asset_return_index_df,
            asset_return_data.asset_return_df_rf)
        asset_alpha_portfolio_performance_data_list[index_name] = asset_alpha_portfolio_performance_data

    # fama-macbeth - time series and cross-sectional        
        asset_fama_macbeth_data = asset_fama_macbeth_data_class(
            work_dir, T, T_momentum, index_name, ticker_list,
            indicator_list = list(asset_indicator_data.parameter_all_indicator_dict.keys()),
            return_df = asset_return_data.asset_return_df, 
            return_rf = asset_return_data.asset_return_df_rf,
            return_index = asset_return_data.asset_return_index_df,
            ff_carhart_factors = asset_factor_data.ff_carhart_factors,
            ticker_indicator_dict = asset_indicator_data.ticker_all_indicator_dict
            )        
        asset_fama_macbeth_data_list[index_name] = asset_fama_macbeth_data
      
    # fama-macbeth - portfolio construction based on fama-macbeth
        asset_fama_macbeth_portfolio = asset_fama_macbeth_portfolio_class(
            work_dir, T, T_momentum, index_name,
            return_df = asset_return_data.asset_return_df, 
            return_rf = asset_return_data.asset_return_df_rf,
            # return_index = asset_return_data.asset_return_index_df,
            betas = asset_fama_macbeth_data.betas_predictive,
            lambda_t = asset_fama_macbeth_data.lambda_t_predictive
            )        
        asset_fama_macbeth_portfolio_list[index_name] = asset_fama_macbeth_portfolio
     
    # now let's do linear regression and portfolio selection
        # portfolio_returns = pd.concat([asset_alpha_portfolio_data.pnl_df,
        #                                asset_fama_macbeth_portfolio.return_fm_portfolio],
        #                               axis=1).dropna(how='any')
        
        asset_alpha_fama_macbeth_portfolio_performance_data = multi_factor_portfolio_performance_class(
            work_dir, T, index_name, 'fama_macbeth',
            # portfolio_returns, 
            asset_fama_macbeth_portfolio.return_fm_portfolio,
            asset_return_data.asset_return_index_df,
            asset_return_data.asset_return_df_rf)
        asset_alpha_fama_macbeth_portfolio_performance_data_list[index_name] = asset_alpha_portfolio_performance_data
#%%