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
S1 = df['BAA'].values - df['AAA'].values
S2 = df['BAA'].values - df['Long'].values
price = df['Price'].values
dividend = df['Dividends'].values
earnings = df['Earnings'].values
cpi = df['CPI'].values
inflation = np.diff(np.log(cpi))
nearngr = np.diff(np.log(earnings))
rearngr = nearngr - inflation
earnyield = earnings/price
nominalPriceRet = np.diff(np.log(price))
realPriceRet = nominalPriceRet - inflation
nominalTotalRet = np.array([np.log(price[k+1] + dividend[k+1]) - np.log(price[k]) for k in range(N)])
realTotalRet = nominalTotalRet - inflation
DFREG = pd.DataFrame({'const' : 1/vol, 'S1' : S1[:-1]/vol, 'S2' : S2[:-1]/vol, 'vol' : 1, 'yield' : np.log(earnyield[:-1])/vol})
allReturns = {'nominal price' : nominalPriceRet, 'real price' : realPriceRet, 'nominal total' : nominalTotalRet, 'real total' : realTotalRet}

for key in allReturns:
    print('Regression for Returns', key, '\n\n')
    returns = allReturns[key]
    Regression = OLS(returns/vol, DFREG).fit()
    print(Regression.summary())
    res = Regression.resid
    plots(res, key)
    analysis(res, key)