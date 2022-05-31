# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 11:50:31 2022

@author: Bill
"""

import pandas as pd # data frames wrangling
import numpy as np # fast and vectorized math functions
import statsmodels.formula.api as smf
import os # operating system operations like check files existance
import matplotlib.pyplot as plt # MATLAB-like plotting library


os.chdir(r'C:\Users\Bill\Desktop\HKU\7037\HW2')
#import all datasets we need
df0 = pd.read_parquet('hw2_option_to_stock_volume_ratio.parquet')
msf = pd.read_parquet('crsp_msf.parquet',columns = ['permno','date','ret','hexcd','prc','vol'])
delist = pd.read_parquet('crsp_msedelist.parquet')
msf = msf.assign(yearmonth=msf['date'].apply(lambda x: str(x)[:7]))

df1 = pd.merge(
    msf, 
    (
        delist
        .assign(yearmonth=delist['dlstdt'].apply(lambda x: str(x)[:7]))
        .loc[delist['dlret'].notnull(), ['permno', 'dlret', 'yearmonth']]
    ),
    how='left',
    on=['permno', 'yearmonth']
)
df1['ret_orig'] = df1['ret']
df1['ret'].update(df1['dlret'])



df2 = pd.read_parquet('comp_finratios.parquet',columns = ['permno','public_date','mktcap'])
#create a common datetime column for merging
df0['YM'] = pd.to_datetime(df0['last_trading_date']).dt.to_period('M')
df1['YM'] = pd.to_datetime(df1['date']).dt.to_period('M')
df2['YM'] = pd.to_datetime(df2['public_date']).dt.to_period('M')
#merge all three datasets
df2 = pd.merge(df0,df2,on=['permno','YM'],sort = True)
df = pd.merge(df1,df2,on = ['permno','YM'],sort = True)
#clean up columns 
df = df.drop(columns= ['date','month_end','first_trading_date', 'last_trading_date','public_date'])
#choose data before 2009
df = df[df.YM < '2009-01']
#create option to stock volume and lagged data column for future use
df['ots'] = df['option_volume']/df['mktcap']
df['ots_lag2'] = df.groupby(['permno'])['ots'].shift(1)
df['prc_lag1'] = df.groupby(['permno'])['prc'].shift(1)
df['mktcap_lag1'] = df.groupby(['permno'])['mktcap'].shift(1)
df['vol_lag1'] = df.groupby(['permno'])['vol'].shift(1)

#remove rows with null value
df = df[df['prc_lag1'].notnull()&
        df['mktcap_lag1'].notnull()&
        df['ots_lag2'].notnull()&
        df['ret'].notnull()&
        df['vol_lag1'].notnull()]
df['key'] = df.index
'''
#add economic restriction - book/market not in the top 20%
top =   (df.groupby('YM').
             apply(lambda x: x.nlargest(int(len(x) * 0.2), 'prc_lag1')).
             droplevel(0).reset_index().sort_index())
'''
#get stocks of bottom 20% trading volume
smallest = (df.groupby('YM').
            apply(lambda x: x.nsmallest(int(len(x) * 0.2), 'vol_lag1')).
            droplevel(0).reset_index().sort_index())

#get stocks of top 50% trading volume
largest =   (df.groupby('YM').
             apply(lambda x: x.nlargest(int(len(x) * 0.5), 'vol_lag1')).
             droplevel(0).reset_index().sort_index())


#get stocks of top 60% mktcap
large = (df.groupby('YM').
         apply(lambda x: x.nlargest(int(len(x) * 0.6), 'mktcap_lag1')).
         droplevel(0).reset_index().sort_index())

#get stocks of bottom 20% mktcap
small = (df.groupby('YM').
         apply(lambda x: x.nsmallest(int(len(x) * 0.2), 'mktcap_lag1')).
         droplevel(0).reset_index().sort_index())
'''
#cut stocks of bottom 20% volume&mktcap and filtered out top 60% mktcap
df = (df[~ ((df.key.isin(large.key))|
         (df.key.isin(smallest.key))|(df.key.isin(small.key)))])#|(df.key.isin(large.key)))] )#(df.key.isin(largest.key))|
       #(df.key.isin(large.key))|
'''                 
#add economic restriction - price higher than 5 dollars and only stocks from NYSE, Arca, Nasdaq
df = df[
        (df['prc_lag1'] >= 1)&        
        (df['hexcd'] < 4)
        ]

#classify stocks of each period by o/s two months ago                       
df['bin'] = (
    df.groupby('YM')    
    .apply(lambda group: np.ceil(group['ots_lag2'].rank() / len(group['ots_lag2']) * 10))
).reset_index(level=[0], drop=True).sort_index()
#creating portfolios according to different levels of o/s
portfolios = (
    df
    .groupby(['YM', 'bin'])
    .apply(
        lambda g: pd.Series({
            'portfolio_ew': g['ret'].mean(),
            'portfolio_vw': (g['ret'] * g['mktcap_lag1']).sum() / g['mktcap_lag1'].sum()
        })
    )
).reset_index()

#check the if there is 10 bins for the first month
portfolios.sort_values(['YM', 'bin']).head(10)

#remove 1996-02 since no enough data to create 10 bins
portfolios = portfolios.drop(portfolios[portfolios.YM == '1996-02'].index)
#portfolio with bin 1&10
portfolios2 = pd.merge(
    portfolios.query('bin==10'),
    portfolios.query('bin==1'),
    suffixes=['_short', '_long'],
    on='YM'
)


#a strategy that short the stocks of top 10% o/s and long stocks of bottom 10% o/s
portfolios2['strategy_vw'] = portfolios2['portfolio_vw_long'] - portfolios2['portfolio_vw_short']
portfolios2['strategy_ew'] = portfolios2['portfolio_ew_long'] - portfolios2['portfolio_ew_short']
portfolios2['strategy_long'] = portfolios2['portfolio_ew_long']
portfolios2['strategy_short'] = portfolios2['portfolio_ew_short']
#getting sharpe ratio and avgret (annualized)
def sharpe(x):
    return x.mean() / x.std()
sharpe_ew = sharpe(portfolios2['strategy_ew'])
avgret_ew = portfolios2['strategy_ew'].mean()
print('annualized sharpe ratio for ew {}'.format(sharpe_ew * 12**0.5))
print('annualized average return for ew {}'.format(avgret_ew*12))
#plot average monthly returns from portfolios of different o/s levels
dta_ew = portfolios.groupby('bin').agg(avgret_ew=('portfolio_ew', 'mean')).sort_values('bin')
plt.rcParams["figure.figsize"] = (8,6)
dta_ew.plot(kind='bar')

#get cumulative return from the strategy and plot the P&L return
portfolios2['cum_ew'] = (portfolios2['strategy_ew'] + 1).cumprod() - 1 # calculates the cumulative return

(
     portfolios2
    .assign(date=portfolios2['YM'])
    .assign(cum_ew=portfolios2['cum_ew']+1)
    .plot(x='date', y='cum_ew').grid(axis='y')
    )

#benchmarking against the fama french factors
ff5 = pd.read_parquet(r'ff_five_factor_monthly.parquet')
ff5['YM'] = pd.to_datetime(ff5['month_end']).dt.to_period('M')

(
    print(smf.ols('strategy_ew ~ 1 + hml + mkt_rf + smb', data=pd.merge(ff5, portfolios2, on='YM'))
    .fit().summary())
)

sharpe_vw = sharpe(portfolios2['strategy_vw'])
avgret_vw = portfolios2['strategy_vw'].mean()
print('annualized sharpe ratio for vw {}'.format(sharpe_vw * 12**0.5))
print('annualized average return for vw {}'.format(avgret_vw*12))
#plot average monthly returns from portfolios of different o/s levels
dta_vw = portfolios.groupby('bin').agg(avgret_vw=('portfolio_vw', 'mean')).sort_values('bin')
plt.rcParams["figure.figsize"] = (8,6)
dta_vw.plot(kind='bar')

#get cumulative return from the strategy and plot the P&L return
portfolios2['cum_vw'] = (portfolios2['strategy_vw'] + 1).cumprod() - 1 # calculates the cumulative return

(
    portfolios2
    .assign(date=portfolios2['YM'])
    .assign(cum_vw=portfolios2['cum_vw']+1)
    .plot(x='date', y='cum_vw').grid(axis='y')
)

#benchmarking against the fama french factors



(
    print(smf.ols('strategy_vw ~ 1 + hml + mkt_rf + smb', data=pd.merge(ff5, portfolios2, on='YM'))
    .fit().summary2())
)

'''
#just long the bin 1

sharpe_long = sharpe(portfolios2['strategy_long'])
avgret_long = portfolios2['strategy_long'].mean()
print('annualized sharpe ratio for long {}'.format(sharpe_long * 12**0.5))
print('annualized average return for long {}'.format((1+avgret_long)**12-1))
#plot average monthly returns from portfolios of different o/s levels



portfolios2['cum_long'] = (portfolios2['strategy_long'] + 1).cumprod() - 1 # calculates the cumulative return

(
    portfolios2
    .assign(date=portfolios2['YM'])
    .assign(cum_long=portfolios2['cum_long']+1)
    .plot(x='date', y='cum_long', logy=True).grid(axis='y')
)

#benchmarking against the fama french factors


(
    print(smf.ols('strategy_long ~ 1 + hml + mkt_rf + smb', data=pd.merge(ff5, portfolios2, on='YM'))
    .fit().summary())
)
# just short
sharpe_short = sharpe(portfolios2['strategy_short'])
avgret_short = portfolios2['strategy_short'].mean()
print('annualized sharpe ratio for short {}'.format(sharpe_short * 12**0.5))
print('annualized average return for short {}'.format((1+avgret_short)**12-1))
#plot average monthly returns from portfolios of different o/s levels



portfolios2['cum_short'] = (portfolios2['strategy_short'] + 1).cumprod() - 1 # calculates the cumulative return

(
    portfolios2
    .assign(date=portfolios2['YM'])
    .assign(cum_short=portfolios2['cum_short']+1)
    .plot(x='date', y='cum_short').grid(axis='y')
)

#benchmarking against the fama french factors


(
    print(smf.ols('strategy_short ~ 1 + hml + mkt_rf + smb', data=pd.merge(ff5, portfolios2, on='YM'))
    .fit().summary())
)

#plot return of ff5
'''

'''
for i in ['mkt_rf', 'smb', 'hml', 'rmw', 'cma']:
    ff5['cum_{}'.format(i)] = (ff5[i] + 1).cumprod() - 1
    (
    ff5
    .assign(date=ff5['YM'])
    .assign(cum=ff5['cum_{}'.format(i)]+1)
    .plot(x='date', y='cum', logy=True).grid(axis='y'))
 #plot strategy against market
 ff5 = pd.read_parquet(r'ff_five_factor_monthly.parquet')
 ff5['YM'] = pd.to_datetime(ff5['month_end']).dt.to_period('M')
 ff5 = ff5[ff5.YM > '1996-02']
 ff5['cum_mkt_rf'] = (ff5['mkt_rf'] + 1).cumprod()  
 fff =  pd.merge(
     portfolios2, 
     (
         ff5
         
     ),
     how='left',
     on=['YM']
 )
 plt.plot(mdates.(ff5['month_end']),ff5['cum_mkt_rf'])
 #plt.plot(mdates.date2num(portfolios2['YM']),portfolios2['cum_ew']+1)
 plt.yscale('log')
''' 
