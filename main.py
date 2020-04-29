import sys
sys.path[0] = "src"
from modules.SR import *
import numpy as np
from matplotlib import pylab as plt
import pandas as pd
from scipy.optimize import leastsq as lsq
from scipy.optimize import curve_fit 
import scipy.stats as spst
from scipy import integrate
from scipy.integrate import quad
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sympy import *
from RegscorePy import *


# 1. Read and Order Data Frame
## 1.2 Read and order data frame
df= pd.read_csv("Mg_Cp.csv", sep=",")
experiments = sorted(df.Ref.unique()) 
colors = plt.cm.rainbow(np.linspace(0, 1, len(experiments)))
for ref, col in zip(experiments, colors):
    sub_df = df[df.Ref==ref]
    plt.scatter(sub_df.Temp, sub_df.Cp, color=col, label=ref)
    
plt.ylim(0, df.Cp.max()*1.1)
plt.xlim(0, df.Temp.max()*1.1)
plt.xlabel("Temperature, in K")
plt.ylabel("Heat capacity, in J/mol*K")
plt.legend(ncol=2, fontsize='small', loc='lower right')
plt.show()
plt.savefig('Read and order data frame.png', dpi = 700)











# # 3. Nonlinear Least Squares Method
## 3.2 Define 'name of regression function' and 'fitting parameters'
new_df = df[df.Temp>5]
func = RW1995_Cp
parmNames = ['ThetaE','a','b']
initialGuess=[236.882352,-0.001703, 0.000018]


# In[85]:


nparm = len(initialGuess)   # number of models parameters
popt,pcov = curve_fit(func, new_df.Temp, new_df.Cp,initialGuess)  # get optimized parameter values and covariance matrix

# Get the parameters
parmEsts = popt
fvec=func(new_df.Temp,*parmEsts)-new_df.Cp   # residuals

# Get the Error variance and standard deviation
RSS = np.sum(fvec**2 )        # RSS = residuals sum of squares
dof = len(new_df) - nparm     # dof = degrees of freedom 
nobs = len(new_df)            # nobs = number of observation
MSE = RSS / dof               # MSE = mean squares error
RMSE = np.sqrt(abs(MSE))           # RMSE = root of MSE

# Get the covariance matrix
cov = pcov

# Get parameter standard errors
parmSE = np.diag( np.sqrt (abs(cov) ) )

# Calculate the t-values
tvals = parmEsts/parmSE

# Get p-values
pvals = (1 - spst.t.cdf( np.abs(tvals),dof))*2

# Get goodnes-of-fit criteria
s2b = RSS / nobs
logLik = -nobs/2 * np.log(2*np.pi) - nobs/2 * np.log(s2b) - 1/(2*s2b) * RSS 


# In[86]:


fit_df=pd.DataFrame(dict( Estimate=parmEsts, StdErr=parmSE, tval=tvals, pval=pvals))
fit_df.index=parmNames











 # 4 Fitting Curve

T_plot = np.linspace(1,1000)
CpRes_RW1995= RW1995_Cp(T_plot,popt[0], popt[1], popt[2])


df= pd.read_csv("Mg_Cp.csv", sep=",")
experiments = sorted(df.Ref.unique()) 
colors = plt.cm.rainbow(np.linspace(0, 1, len(experiments)))
for ref, col in zip(experiments, colors):
    sub_df = df[df.Ref==ref]
    plt.scatter(sub_df.Temp, sub_df.Cp, color=col, label=ref)
plt.plot(T_plot, CpRes_RW1995,label="CpRes_RW1995")
plt.ylim(0, df.Cp.max()*1.1)
plt.xlim(0, df.Temp.max()*1.1)
plt.xlabel("Temperature, in K")
plt.ylabel("Heat capacity, in J/mol*K")
plt.legend(ncol=3, fontsize='small', loc='best', bbox_to_anchor=(1, 1))
plt.show()
plt.savefig('Fitting Curve.png', dpi = 700)

