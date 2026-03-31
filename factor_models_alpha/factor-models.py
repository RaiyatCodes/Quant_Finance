#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
from datetime import datetime, timedelta
import re
import pingouin as pg
import statsmodels.api as sm

import requests
# import pytz
from bs4 import BeautifulSoup
# import random
# import csv
from concurrent.futures import ThreadPoolExecutor

from Portfolio_Optimization_class import *

## class for all asset data
class asset_list_data_class:
    def __init__(self, file_dir, index_list):
        self.file_dir = file_dir
        self.index_list = index_list
        
        # to prevent you from being identifed as a bot
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/115.0 Safari/537.36"
        }
        
        # list of assets        
        index_dict = {
            x: (f"List_of_S%26P_{x[2:]}_companies" if x.startswith("sp") else "Dow_Jones_Industrial_Average")
            for x in index_list
        }
        
        self.stock_list_list = []
        self.stock_ranked_list_list = {}
        self.stock_ranked_dict_list = {}
        for index_name in index_dict.keys():
            print (f"getting stock list in {index_name}...")
            # url of stock list on Wikipedia
            url_index = f'https://en.wikipedia.org/wiki/{index_dict[index_name]}'
        
            if (index_name == 'sp500')|(index_name == 'sp400')|(index_name == 'sp600'):
                stock_list = self.get_stock_list_sp(index_name, url_index)           
            elif index_name == 'djia':
                stock_list = self.get_stock_list_djia(index_name, url_index)
            else:
                print('index name needs to be either sp500, sp400, sp600 or djia')            
            self.stock_list_list.append(stock_list)
                
            stock_ranked_list, stock_ranked_dict = self.get_top_stocks(stock_list)
            self.stock_ranked_list_list[index_name] = stock_ranked_list
            self.stock_ranked_dict_list[index_name] = stock_ranked_dict
        
    def get_stock_list_sp(self, index_name, url):
        file_dir = self.file_dir
        headers = self.headers
        # Fetch the webpage        
        response = requests.get(url, headers=headers)
        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")
        # Find the table containing the index constituents
        table = soup.find("table", {"class": "wikitable"})
    
        # Extract the data from the table into a pandas DataFrame
        data = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            data.append((cols[0].text.strip(), cols[1].text.strip(), cols[2].text.strip(), cols[3].text.strip()))
    
        # Create a DataFrame from the extracted data
        stock_list = pd.DataFrame(data, columns=["Symbol", "Company Name", "GICS Sector", "GICS Sub-Industry"])
        stock_list = stock_list[stock_list['Symbol']!='CON'].reset_index(drop=True).set_index('Symbol')
        stock_list.name = index_name
                
        file_name = file_dir + index_name + '_list.csv'
        stock_list.to_csv(file_name, index=True)
        
        return stock_list
    
    def get_top_stocks(self, stock_list):
        file_dir = self.file_dir
        print(f'getting stock info for {stock_list.name}')

        # Get the list of tickers
        index_name = stock_list.name
        tickers = stock_list.index.tolist()
    
        # Function to fetch market cap Yahoo
        # def get_market_cap(ticker):
        def get_stock_info(ticker):
            try:
                # print(f'ticker = {ticker}')
                stock = yf.Ticker(ticker)
                # extract values
                ticker_data = {
                    "market cap": stock.info.get("marketCap"),
                    "current price": stock.info.get("currentPrice"),
                    "shares outstanding": stock.info.get("sharesOutstanding"),
                    "book value": stock.info.get("bookValue"),
                    "book_price_ratio": stock.info.get("bookValue")/stock.info.get("currentPrice"),
                    "52 week change": stock.info.get("52WeekChange"),
                }
                
                # make DataFrame (1 row, columns are names, index is ticker)
                ticker_data_df = pd.DataFrame([ticker_data], index=[ticker])                                
                return ticker_data_df
            except: # KeyError:
                print(f'cannot fetch {ticker} data from yahoo.')
                return None
                     
        # Fetch market caps for all tickers
        stock_info_list = {ticker: get_stock_info(ticker) for ticker in tickers}        
        stock_info_list = {k: v.iloc[0] for k, v in stock_info_list.items() if v is not None}
        
        # Convert to DataFrame
        stock_info_list = pd.DataFrame.from_dict(stock_info_list, orient="index")
        stock_info_list = stock_info_list.dropna().sort_values(by='market cap', ascending=False)
        stock_info_list.index.name = 'Symbol'
    
        # Merge with the original table to get company names
        top_stocks = stock_info_list.merge(stock_list, left_index=True, right_on='Symbol')#.reset_index(drop=True)
        top_stocks.name = index_name
        
        # top_stocks_dict = top_stocks["Company Name"].to_dict()
        top_stocks_dict = dict(zip(top_stocks.index, top_stocks["Company Name"]))
        
        top_stocks = top_stocks.set_index("Company Name", drop=False)
        file_name = file_dir + index_name + '_ranked_list.csv'
        top_stocks.to_csv(file_name, index=True)
       
        return top_stocks, top_stocks_dict
        
    def get_stock_list_djia(self, index_name, url):
        # Fetch the webpage
        file_dir = self.file_dir
        headers = self.headers
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Ensure we notice bad responses        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")       
        # Find the table containing the DJIA constituents
        table = soup.find("table", {"class": "wikitable"})
        #print(table.prettify())
        
        # Extract the data from the table into a pandas DataFrame
        data = []
        for row in table.find_all("tr")[1:]:
            cols_name = row.find_all("th")
            cols = row.find_all("td")
            #company_name = cols[0].text.strip()  # Company name is in the first column
            company_name = cols_name[0].text.strip()  # Company name is in the first column
            symbol = cols[1].text.strip()        # Symbol is in the second column
            industry = cols[2].text.strip()
            index_weight = cols[5].text.strip()
            # data.append((symbol, company_name))
            data.append((symbol, company_name, industry, index_weight))
        
        # Create a DataFrame from the extracted data
        # stock_list = pd.DataFrame(data, columns=["Symbol", "Company Name"])
        stock_list = pd.DataFrame(data, columns=["Symbol", "Company Name", "Industry", "Index_Weight"]).set_index('Symbol')
        stock_list.name = index_name
        
        file_name = file_dir + 'djia_list.csv'
        stock_list.to_csv(file_name, index=True)
        
        return stock_list

class asset_single_factor_simple_class:
    def __init__(self, file_dir, T, index_name, asset_return_data):
        self.file_dir = file_dir
        self.T = T
        self.p_lim = 0.05
        
        self.index_name = index_name
        self.asset_return_class = asset_return_data        
        self.return_df = asset_return_data.asset_return_df
        self.return_rf = asset_return_data.asset_return_df_rf
        self.return_index = asset_return_data.asset_return_index_df
        
        self.ticker_regresion_list = self.single_index_model_regressions()
        self.sml_plots = self.sml_data_plots()

    def single_index_model_regressions(self):
        file_dir = self.file_dir
        p_lim = self.p_lim
        return_df = self.return_df # return data for all qualified stocks in index
        return_index = self.return_index # return data for the relevant index
        return_rf = self.return_rf # return data for risk fee asset
                
        # return_df = asset_single_factor.return_df
        # return_index = asset_single_factor.return_index
        # return_rf = asset_single_factor.return_rf
        
        return_index = return_index.loc[return_df.index]
        
        # set up the linear regressions
        return_df_ex = return_df - return_rf.values
        return_index_ex = return_index - return_rf.values
    
        ticker_regression_list = []
        # loop through the assets
        for return_ticker in return_df_ex.columns:
            Y = return_df_ex[return_ticker]
            X = return_index_ex

            y = Y
            x = sm.add_constant(X)
            model = sm.OLS(y, x).fit()
            
            # residual               
            # collect results in dict
            dt_results = {
                "alpha": model.params["const"],
                "alpha_p": model.pvalues["const"],
                "alpha_se": model.bse["const"],                    
                "alpha_t": model.tvalues["const"],

                "beta": model.params[1],
                "beta_p": model.pvalues[1],
                "beta_se": model.bse[1],
                "beta_t": model.tvalues[1],
                
                "R2": model.rsquared,
                "Adj_R2": model.rsquared_adj,
                
                "ResVar": np.var(model.resid, ddof=1), # unbiased variance of residuals
                "ResStd": np.std(model.resid, ddof=1), # unbiased std of residuals
                
                "DW": sm.stats.stattools.durbin_watson(model.resid),
                "Nobs": int(model.nobs),
                "Fstat": model.fvalue,
                "F_p": model.f_pvalue,
                
                f'u_m_{X.columns[0]}': X.mean().iloc[0],
                f'var_m_{X.columns[0]}': X.var().iloc[0]
            }
            
            # 1×n DataFrame
            dt_results = pd.DataFrame([dt_results], index=[return_ticker])
            dt_results['alpha_sig'] = dt_results['alpha'].values[0] if dt_results['alpha_p'].values[0]<=p_lim else 0.0
            dt_results['beta_sig'] = dt_results['beta'].values[0] if dt_results['beta_p'].values[0]<=p_lim else 0.0
            u_v_column = f'u/var_m_{X.columns[0]}'
            dt_results[u_v_column] = dt_results[f'u_m_{X.columns[0]}'] / dt_results[f'var_m_{X.columns[0]}']
            ticker_regression_list.append(dt_results)

            # and the regression plots
            # plot scatter
            plt.scatter(X, Y, color="blue", label="scatter plot", s=5)            
            # Plot regression line
            plt.plot(X, model.params[1]*X + model.params["const"], color="red", label="linear regression plot")
            
            # Labels and legend
            plot_title = f'Regression_Plot_with_{Y.to_frame().columns[0]}_vs_{X.columns[0]}'
            plt.title(plot_title)
            plt.xlabel(X.columns[0])
            plt.ylabel(Y.to_frame().columns[0])
            plt.legend()
                        
            # Save to file
            plot_title = plot_title.replace("/", "_")
            plt.savefig(file_dir+plot_title+'.png', dpi=300, bbox_inches="tight")           
            plt.show()    
            
            # and the return plots
            # plot return of stock
            plt.plot(Y, color="blue", label=f"{Y.to_frame().columns[0]} return")            
            # plot return of index
            plt.plot(X, color="red", label=f"{X.columns[0]} return" )
            
            # Labels and legend
            plot_title = f'Return_Plot_with_{Y.to_frame().columns[0]}_and_{X.columns[0]}'
            plt.title(plot_title)
            plt.xlabel("date")
            plt.ylabel("return")
            plt.legend()
            
            # Save to file
            plot_title = plot_title.replace("/", "_")
            plt.savefig(file_dir+plot_title+'.png', dpi=300, bbox_inches="tight")           
            plt.show()  
                        
        ticker_regression_list = pd.concat(ticker_regression_list, axis=0)
        ticker_regression_list.to_csv(file_dir+f'{return_index_ex.columns[0]}_regression_results.csv')
        
        return ticker_regression_list

    def sml_data_plots(self):
        file_dir = self.file_dir
        T = self.T
        index_name = self.index_name
                
        return_df = self.return_df # return data for all qualified stocks in index
        return_index = self.return_index # return data for the relevant index
        return_rf = self.return_rf # return data for risk fee asset
        ticker_regression_list = self.ticker_regresion_list
        
        # return_df = asset_single_simple_factor.return_df
        # return_index = asset_single_simple_factor.return_index
        # return_rf = asset_single_simple_factor.return_rf
        # ticker_regression_list = asset_single_simple_factor.ticker_regresion_list
        
        return_index = return_index.loc[return_df.index]        
        # set up the linear regressions
        return_df_ex = return_df - return_rf.values
        return_index_ex = return_index - return_rf.values
        
        return_index_ex_total = (1 + return_index_ex).prod()
        return_index_ex_annualized = return_index_ex_total**(T/len(return_index_ex)) - 1
        
        return_rf_total = (1 + return_rf).prod()
        return_rf_annualized = return_rf_total**(T/len(return_rf)) - 1
                
        beta_list = ticker_regression_list['beta_sig']
        capm_return_df_ex = return_index_ex[index_name].values.reshape(-1, 1) * beta_list.values.reshape(1, -1)
        capm_return_df_ex = pd.DataFrame(capm_return_df_ex, 
                                         index=return_index_ex.index, 
                                         columns=beta_list.index)
                
        for return_ticker in capm_return_df_ex.columns:
            # --- Inputs ---
            beta = beta_list[return_ticker]            
            beta_grid = np.linspace(0, beta*2, 200)
            
            Rf = return_rf_annualized.values[0] #0.0   # risk-free rate (intercept)
            MRP = return_index_ex_annualized.values[0]  # market risk premium (slope)
            
            # SML line
            sml_line = Rf + MRP * beta_grid
            
            # Stock's expected return at its beta
            er_stock = Rf + MRP * beta
            
            # CAPM expected returns time series for this stock
            capm_df = Rf + capm_return_df_ex[return_ticker]
            
            # --- Plot ---
            plt.figure(figsize=(8,6))
            plt.plot(beta_grid, sml_line,
                     label=f"SML: E[R] = {Rf:.2%} + {MRP:.2%}·β",
                     color="black")
            
            # Mark the theoretical CAPM point
            plt.scatter([beta], [er_stock], marker='x', color="red", s=20, zorder=5, label="CAPM (theoretical)")
            
            # Add CAPM time series (vertical stack at this beta)
            plt.scatter([beta]*len(capm_df), capm_df.values, marker='s',
                        color="blue", s=5, alpha=0.5, label="CAPM time series")
            
            # Projection lines from axes up to the CAPM theoretical point
            plt.plot([beta, beta], [0, er_stock], color="red", linestyle="--", alpha=0.7)
            plt.plot([0, beta], [er_stock, er_stock], color="red", linestyle="--", alpha=0.7)
            
            # Labels near axes
            x_offset = (plt.xlim()[1] - plt.xlim()[0]) * 0.01
            y_offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.01
            
            plt.text(beta + x_offset, 0 + y_offset, f"β={beta:.2f}",
                     ha="left", va="bottom", color="black")
            plt.text(0 + x_offset, er_stock + y_offset, f"ER={er_stock:.2%}",
                     ha="left", va="bottom", color="black")
            
            plt.xlabel("Beta (β)")
            plt.ylabel("Expected Return")
            
            plot_title = f'SML_{return_ticker}_{index_name}'
            plt.title(plot_title)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            
            # Save to file
            plot_title = plot_title.replace("/", "_")
            plt.savefig(file_dir+plot_title+'.png', dpi=300, bbox_inches="tight")                       
            plt.show()
        
        return True

class asset_single_factor_class:
    def __init__(self, file_dir, T, index_name, asset_return_data):
        self.file_dir = file_dir
        self.T = T
        self.p_lim = 0.05
        self.shift = 1
        self.index_name = index_name
        self.asset_return_class = asset_return_data        
        self.return_df = asset_return_data.asset_return_df
        self.return_rf = asset_return_data.asset_return_df_rf
        self.return_index = asset_return_data.asset_return_index_df
        
        self.ticker_dates_dict, self.return_dates_dict = self.single_index_model_regressions()
        self.active_portfolio_data, self.weight_adj_dates, self.weight_adj_index_dates = self.single_index_model_weight()
        
        self.return_portfolio, self.return_portfolio_active_index = self.sim_portfolio_performance()
        
    def single_index_model_regressions(self):
        file_dir = self.file_dir
        T = self.T
        index_name = self.index_name
        p_lim = self.p_lim
        return_df = self.return_df # return data for all qualified stocks in index
        return_index = self.return_index # return data for the relevant index
        return_rf = self.return_rf # return data for risk fee asset
        
        lookback = T            
        freq = T #if cov_by_date else 1.0                                                                  
        
        # return_df = asset_single_factor.return_df
        # return_index = asset_single_factor.return_index
        # return_rf = asset_single_factor.return_rf
        
        return_index = return_index.loc[return_df.index]
        
        # set up the linear regressions
        return_df_ex = return_df - return_rf.values
        return_index_ex = return_index - return_rf.values
        
        # rolling window is of T, 252 days
        dates = return_df.index[T-1:]
    
        ticker_dates_dict = {}
        # loop through the assets
        for return_ticker in return_df_ex.columns:
            Y = return_df_ex[return_ticker]
            X = return_index_ex
            
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
                
                # residual               
                # collect results in dict
                dt_results = {
                    "alpha": model.params["const"],
                    "alpha_p": model.pvalues["const"],
                    "alpha_se": model.bse["const"],                    
                    "alpha_t": model.tvalues["const"],

                    "beta": model.params[1],
                    "beta_p": model.pvalues[1],
                    "beta_se": model.bse[1],
                    "beta_t": model.tvalues[1],
                    
                    "R2": model.rsquared,
                    "Adj_R2": model.rsquared_adj,
                    
                    "ResVar": np.var(model.resid, ddof=1), # unbiased variance of residuals
                    "ResStd": np.std(model.resid, ddof=1), # unbiased std of residuals
                    
                    "DW": sm.stats.stattools.durbin_watson(model.resid),
                    "Nobs": int(model.nobs),
                    "Fstat": model.fvalue,
                    "F_p": model.f_pvalue,
                    
                    'u_m': (X.iloc[end_pos-lookback+1: end_pos+1]).mean().iloc[0],
                    'var_m': (X.iloc[end_pos-lookback+1: end_pos+1]).var().iloc[0]
                }
                
                # 1×n DataFrame
                dt_results = pd.DataFrame([dt_results], index=[dt])
                dt_results['alpha_sig'] = dt_results['alpha'].values[0] if dt_results['alpha_p'].values[0]<=p_lim else 0.0
                dt_results['beta_sig'] = dt_results['beta'].values[0] if dt_results['beta_p'].values[0]<=p_lim else 0.0
                dt_results['u/var_m'] = dt_results['u_m'] / dt_results['var_m']

                ticker_dt_list.append(dt_results)
                
            ticker_dt_list = pd.concat(ticker_dt_list, axis=0)
            ticker_dt_list.name = return_ticker
            ticker_dates_dict[return_ticker] = ticker_dt_list
            ticker_dt_list.to_csv(file_dir+index_name+'_'+return_ticker.replace('/','_')+'_ticker_sim.csv')
        
                  
        panel = pd.concat(ticker_dates_dict, axis=1)  # MultiIndex columns: (ticker, old_col)
        fields = panel.columns.get_level_values(1).unique()       
        # return_dates_dict = {f: panel.xs(f, axis=1, level=1) for f in fields}        
        return_dates_dict = {}
        for f in fields:
            return_dates_dict[f] = panel.xs(f, axis=1, level=1)
            f_name = f.replace("/", "_")
            return_dates_dict[f].to_csv(file_dir+index_name+'_'+f_name+'_parameter_sim.csv')
        
        return ticker_dates_dict, return_dates_dict

    def single_index_model_weight(self):
        return_dates_dict = self.return_dates_dict
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

        # alpha of active portfolio
        alpha_active_portfolio = weight_0_dates.loc[:, weight_0_dates.columns != "total"] * alpha_0_dates
        alpha_active_portfolio['total'] = alpha_active_portfolio.sum(axis=1)
        alpha_active_portfolio = alpha_active_portfolio['total']

        # residue variance of active portfolio        
        var_active_portfolio = weight_0_dates.loc[:, weight_0_dates.columns != "total"]**2 * var_0_dates
        var_active_portfolio['total'] = var_active_portfolio.sum(axis=1)
        var_active_portfolio = var_active_portfolio['total']
        
        # wA0: initial positions in active portfolio
        weight_active_portfolio_0 = (alpha_active_portfolio/var_active_portfolio).fillna(0)
        return_var_m = return_dates_dict['u/var_m'].copy(deep=True).iloc[:,0]
        weight_active_portfolio_0 = weight_active_portfolio_0/ return_var_m.values
        # weight_active_portfolio_0 = weight_active_portfolio_0.clip(lower=-weight_lim, upper=weight_lim) # avoid divided by 0
        
        # beta
        beta_0_dates = return_dates_dict['beta_sig'].copy(deep=True)
        beta_active_portfolio = weight_0_dates.loc[:, weight_0_dates.columns != "total"] * beta_0_dates 
        beta_active_portfolio['total'] = beta_active_portfolio.sum(axis=1)
        beta_active_portfolio = beta_active_portfolio['total']
        
        # wAa: adjusted positions in active portfolio
        weight_active_portfolio_adj =  (weight_active_portfolio_0 / (1+(1-beta_active_portfolio)*weight_active_portfolio_0)).clip(lower=0.0, upper=weight_lim)

        active_portfolio_data = pd.concat([alpha_active_portfolio, 
                                          var_active_portfolio,
                                          beta_active_portfolio,                                  
                                          weight_active_portfolio_0,
                                          weight_active_portfolio_adj],
                                          axis=1)
        active_portfolio_data.columns = pd.Index(['alpha', 'var', 'beta', 'weight_act_0', 'weight_act_adj'])
        active_portfolio_data.loc[active_portfolio_data['weight_act_0']<0,'weight_act_adj'] = weight_lim
        active_portfolio_data[index_name] = 1.0 - active_portfolio_data['weight_act_adj']

        # active_dates = active_portfolio_data[active_portfolio_data['weight_act_adj']!=0].index
        weight_adj_dates = weight_0_dates.mul(active_portfolio_data['weight_act_adj'], axis=0)
        weight_adj_index_dates = pd.concat([weight_adj_dates[weight_adj_dates.columns[weight_adj_dates.columns!='total']],
                                            active_portfolio_data[index_name]], axis=1)
        weight_adj_index_dates['total'] = weight_adj_index_dates.sum(axis=1)  
        # weight_adj_index_light_dates = weight_adj_index_dates.loc[:, (weight_adj_index_dates != 0).any(axis=0)]
        
        active_portfolio_data.to_csv(file_dir+index_name+'_active_portfolio_parameters.csv')
        weight_adj_dates.to_csv(file_dir+index_name+'_weight_adj_no_index.csv')
        weight_adj_index_dates.to_csv(file_dir+index_name+'_weight_adj_with_index.csv')
        
        return active_portfolio_data, weight_adj_dates, weight_adj_index_dates

    def sim_portfolio_performance(self):
        file_dir = self.file_dir
        T = self.T
        shift = self.shift
        index_name = self.index_name
        p_lim = self.p_lim
        return_df = self.return_df # return data for all qualified stocks in index
        return_index = self.return_index # return data for the relevant index
        return_rf = self.return_rf # return data for risk fee asset
        weight_adj_index_dates = self.weight_adj_index_dates.copy(deep=True)
        
        # return_df = asset_single_factor.return_df
        # return_index = asset_single_factor.return_index
        # return_rf = asset_single_factor.return_rf
        # weight_adj_index_dates = asset_single_factor.weight_adj_index_dates.copy(deep=True)

        return_index = return_index.loc[return_df.index]
        weight_adj_index_dates = weight_adj_index_dates[weight_adj_index_dates.columns[weight_adj_index_dates.columns!='total']]
        
        return_df_index = pd.concat([return_df, return_index], axis=1)
        
        return_portfolio = (weight_adj_index_dates.shift(shift) * return_df_index).dropna(how='all')
        return_portfolio['total'] = return_portfolio.sum(axis=1) 
        return_portfolio_active_index = return_portfolio['total'].to_frame()
        return_portfolio_active_index.columns = pd.Index([f'active_{index_name}'])
        
        return_portfolio.to_csv(file_dir+index_name+'_portfolio_return_all.csv')
        return_portfolio_active_index.to_csv(file_dir+index_name+'_portfolio_return_active_index.csv')
                                                                         
        return return_portfolio, return_portfolio_active_index

class sim_portfolio_performance_class:
    def __init__(self, file_dir, T, portfolio_performance_df,return_index_df,return_rf):
        self.start = 100.0 # initial portfolio wealth
        self.p_lim = 0.05
        self.file_dir = file_dir
        self.T = T
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
        portfolio_performance_df = self.portfolio_performance_df
        return_index_df = self.return_index_df
        start = self.start
        
        # file_dir = portfolio_performance.file_dir
        # portfolio_performance_df = portfolio_performance.portfolio_performance_df
        # return_index_df = portfolio_performance.return_index_df
        # start = portfolio_performance.start       
        
        returns_df = pd.concat([portfolio_performance_df, return_index_df], axis=1)
        
        # Start all price series at 100        
        prices_df = (1 + returns_df).cumprod() * start
        prices_df.to_csv(file_dir+'portfolio_price.csv', index=True, index_label='date')
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
        
    def sim_return_data(self):
        return_df = self.portfolio_performance_df
        return_index = self.return_index_df
        return_rf = self.return_rf
        p_lim = self.p_lim
        
        # return_df = portfolio_performance.portfolio_performance_df
        # return_index = portfolio_performance.return_index_df
        # return_rf = portfolio_performance.return_rf    
        
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
        
        tbl.to_csv(file_dir+'single_index_model_portfolio_performance.csv')
        
        return tbl

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
        returns_df = self.portfolio_performance_df
        
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
