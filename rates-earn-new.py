import pandas as pd
import numpy as np
from statsmodels.api import OLS
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
from statsmodels.tsa.stattools import acf

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.savefig('O-' + label + '.png')
    plt.close()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.savefig('A-' + label + '.png')
    plt.close()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.savefig('QQ-' + label + '.png')
    plt.close()
    
def analysis(data, label):
    print(label + ' analysis of residuals normality')
    print('Skewness:', round(stats.skew(data), 3))
    print('Kurtosis:', round(stats.kurtosis(data), 3))
    print('Shapiro-Wilk p = ', round(100*stats.shapiro(data)[1], 1), '%')
    print('Jarque-Bera p = ', round(100*stats.jarque_bera(data)[1], 1), '%')
    print('Autocorrelation function analysis for ' + label)
    L1orig = sum(abs(acf(data, nlags = 5)[1:]))
    print('\nL1 norm original residuals ', round(L1orig, 3), label, '\n')
    L1abs = sum(abs(acf(abs(data), nlags = 5)[1:]))
    print('L1 norm absolute residuals ', round(L1abs, 3), label, '\n')

df = pd.read_excel("rates-earn-new.xlsx", sheet_name = 'data')
vol = df["Volatility"].values[1:]
N = len(vol)
print('Data size = ', N)

S1 = df["BAA"].values - df["AAA"].values
S2 = df['BAA'].values - df['Long'].values
DF0REG = pd.DataFrame({'const' : 1/vol, 'vol' : 1})
DF1REG = pd.DataFrame({'const' : 1, 'S1' : S1[:-1], 'S2' : S2[:-1]})
DF2REG = pd.DataFrame({'const' : 1/vol, 'S1' : S1[:-1]/vol, 'S2' : S2[:-1]/vol})
DF3REG = pd.DataFrame({'const' : 1/vol, 'vol' : 1, 'S1' : S1[:-1]/vol, 'S2' : S2[:-1]/vol})

price = df['Price'].values
dividend = df['Dividends'].values
earnings = df['Earnings'].values
cpi = df['CPI'].values
inflation = np.diff(np.log(cpi))
nearngr = np.diff(np.log(earnings))
rearngr = nearngr - inflation
earnyield = earnings/price
ngrowth = nearngr/vol
nmeangrowth = np.mean(ngrowth)
rgrowth = rearngr/vol
rmeangrowth = np.mean(ngrowth)
DFs = {'vol-const': DF0REG, 'spreads' : DF1REG, 'vol-spreads' : DF2REG, 'vol-spreads-const' : DF3REG}
for item in DFs:
    RegNGrowth = OLS(ngrowth, DFs[item]).fit()
    print('Nominal Earnings Growth', item)
    print(RegNGrowth.summary())
    resngrowth = RegNGrowth.resid
    plots(resngrowth, 'Nominal Earnings Growth '+ item)
    analysis(resngrowth, 'Nominal Earnings Growth ' + item)
    RegRGrowth = OLS(rgrowth, DFs[item]).fit()
    print('Real Earnings Growth', item)
    print(RegRGrowth.summary())
    resrgrowth = RegRGrowth.resid
    plots(resrgrowth, 'Real Earnings Growth ' + item)
    analysis(resrgrowth, 'Real Earnings Growth ' + item)
