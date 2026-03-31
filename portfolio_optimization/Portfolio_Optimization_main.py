#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For ECON369 Spring 2026 Assignment 1
@author: raiyat
"""

# first set up home directory
import os
import numpy as np
import pandas as pd
from datetime import datetime

### main function
if __name__ == "__main__": 
    file_dir = "/Users/raiyat/Documents/Documents/AC - Gettysburg/2026-Spring/ECON369/Assignment_1/"    
    os.makedirs(file_dir, exist_ok=True)
    os.chdir(file_dir)
    os.getcwd()
    
    # import our own classes
    from Portfolio_Optimization_class import *
    
    T = 252 # average number of trading days for us market
    start_date = datetime(2010, 1, 1).date()
    end_date = datetime(2030, 12, 31).date()
    
    # list of assets
    # Fantastic Seven
    ticker_list = { "NVDA": "Nvidia",
                    "MSFT": "Microsoft",
                    "AAPL": "Apple",
                    "GOOGL": "Alphabet (A)",
                    "AMZN": "Amazon",
                    "META": "Meta",
                    "TSLA": "Tesla"}  
    
    # ticker_list = {"COST": "Costco",
    #                "KO": "Coca-Cola"}
    
    ticker_rf = 'SOFR'
    ticker_index_list = {'^GSPC':'SP500'}
    
    # all asset data in class
    is_portfolio = False
    asset_return_data = asset_return_data_class(
        file_dir, T, is_portfolio,
        ticker_list, ticker_rf, ticker_index_list,
        start_date, end_date
        )
#%%
    # all asset frontier in class
    # freq = 1 #252  
    # freq = 252
    # lookback = 252   
    cap = 0.5 # cap limit for 1 asset
    risky_asset_limit = 0.75 #1.0
    rf_weight = 1 - risky_asset_limit
    # Attach λ-sweep for each date (λ = 0.5, 1.0, …, 5.0) or non-linear spacing
    # lambdas = np.array([0.001, 0.01, 0.1, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    lambdas = np.array([2, 5])    
    # lambdas = None
    cov_is_daily=True

    efficient_frontier_data = efficient_frontier_class(
        file_dir, T, cap, rf_weight, lambdas,
        asset_return_data.asset_return_df, 
        asset_return_data.asset_return_df_cov_list, 
        asset_return_data.asset_return_df_rf,         
        cov_is_daily)
#%%
    # all asset trade
    portfolio_optimization = portfolio_optimization_class(
        file_dir, efficient_frontier_data.portfolio_df_dict,
        asset_return_data.asset_return_combined, 
        asset_return_data.asset_return_df_rf,  
        asset_return_data.asset_return_df_cov_list
        )
#%%  
    # now let's look at portfolio performance and compare
    is_portfolio = True
    portfolio_performance = portfolio_performance_class(
        file_dir, T,
        portfolio_optimization.portfolio_performance_dict,
        asset_return_data.asset_return_index_df,
        asset_return_data.asset_return_df_rf
        )
#%% 