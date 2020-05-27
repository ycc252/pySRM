#!/usr/bin/env python
# coding: utf-8

# # 1. Read and Order Data Frame

# ## 1.1 Import packages

# In[63]:


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


# ## 1.2 Read and order data frame

# In[64]:


df= pd.read_csv("Mg_Cp.csv", sep=",")
experiments = sorted(df.Ref.unique()) 
colors = plt.cm.rainbow(np.linspace(0, 1, len(experiments)))
# for ref, col in zip(experiments, colors):
#     sub_df = df[df.Ref==ref]
#     plt.scatter(sub_df.Temp, sub_df.Cp, color=col, label=ref)
    
# plt.ylim(0, df.Cp.max()*1.1)
# plt.xlim(0, df.Temp.max()*1.1)
# plt.xlabel("Temperature, in K")
# plt.ylabel("Heat capacity, in J/mol*K")
# plt.legend(ncol=2, fontsize='small', loc='lower right')
# plt.show()
# plt.savefig('Read and order data frame.png', dpi = 300)

# # 2. Models

# ## 2.1 Debye models

# In[65]:


def integrand(y):
    return (y**4 * np.exp(y)) /((np.exp(y) - 1)**2)
def Debye_Cp(T,*param):
    Theta_D = param[0]
    Cp_debye =[]
    for i in T:
        x =i/Theta_D
        Cp_debye.append( 9 * 8.314 * (x**3) * quad(integrand, 0, 1/x)[0] )    
    return Cp_debye


# ## 2.2 Einstein Model

# In[66]:


def Einstein_Cp(T,*param):
    Theta_E = param[0]
    Cp_EM_final = []
    Cp_Einstein = 3.0 * 8.314 * (Theta_E/T)**2.0 * (np.exp(Theta_E/T)/(np.exp(Theta_E/T)-1)**2.0)  
    return Cp_Einstein


# ## 2.3 Bent Cable model

# In[67]:


def bcm_Cp(T,*param):
    beta1 = param[0]
    beta2 = param[1]
    alpha = param[2]
    gamma = param[3]
    Cp_bcm = []
    for i in T:
        if i < (alpha - gamma):
            Cp_bcm .append(beta1 * i)
        elif i > (alpha+gamma) :
            Cp_bcm .append((beta1 * i) + (beta2 * (i - alpha)))
        else :
            Cp_bcm .append(beta1 * i + beta2 * (i - alpha + gamma)**2/(4*gamma))
    return Cp_bcm


# ## 2.4 Ringberg Workshop’1995 model

# In[68]:


def RW1995_Cp(T,*param):
    Theta_E = param[0]
    a = param[1]
    b = param[2]
    Cp_RW1995 = []
    Cp_RW1995 = Einstein_Cp(T, *param)  + a * T + b * T**2   
    return Cp_RW1995


# ## 2.5 Chen-Sundman model 

# In[69]:


def CS_Cp(T,*param):
    Theta_E = param[0]
    a = param[1]
    b = param[2]
    Cp_CS = []
    Cp_CS = Einstein_Cp(T, *param)  + a * T + b * T**4   
    return Cp_CS


# ## 2.6 Generalized Einstein-polynomial model

# In[70]:


def GEPM_Cp(T,*param):
    Theta_E = param[0]
    a = param[1]
    b = param[2]
    c = param[3]
    d = param[4]
    Cp_GEPM = []
    Cp_GEPM = Einstein_Cp(T, *param)  + a * T + b * T**2 + c * T**3 + d * T**4   
    return Cp_GEPM


# ## 2.7 Segmented Regression

# ### 2.7.1 SRD = Debye + BCM

# In[81]:


def Cp_SR_Debye(T,*param):
    Cp_SR_D = []
    Parameter_Debye = []
    Parameter_Debye = param[0]
    Parameter_BCM = []
    Parameter_BCM.append(param[1])
    Parameter_BCM.append(param[2])
    Parameter_BCM.append(param[3])
    Parameter_BCM.append(param[4])
    Cp_SR_De = Debye_Cp(T,Parameter_Debye)
    Cp_SR_bc= bcm_Cp(T,*Parameter_BCM)
    for i in range (len(Cp_SR_De)):
        Cp_SR_D_i = Cp_SR_De[i] + Cp_SR_bc[i]
        Cp_SR_D .append(Cp_SR_D_i)
    return Cp_SR_D


# ### 2.7.2 SRE = Einstein + BCM

# In[82]:


def Cp_SR_Einstein(T,*param):
    Cp_SR_E = []
    Parameter_Einstein = []
    Parameter_Einstein =param[0]
    Parameter_BCM = []
    Parameter_BCM.append(param[1])
    Parameter_BCM.append(param[2])
    Parameter_BCM.append(param[3])
    Parameter_BCM.append(param[4])
    Cp_SR_Ei = Einstein_Cp(T,Parameter_Einstein)
    Cp_SR_bc= bcm_Cp(T,*Parameter_BCM)
    for i in range (len(Cp_SR_Ei)):
        Cp_SR_E_i = Cp_SR_Ei[i] + Cp_SR_bc[i]
        Cp_SR_E .append(Cp_SR_E_i)
    return Cp_SR_E 

# ## 2.8 Linear Combination of Einstein functions

# In[11]:


def Einstein_Cp2(T,*param):
    Theta_E2 = param[0]
    Cp_Einstein2 = []
    Cp_Einstein2 = 3.0 * 8.314 * (Theta_E2/T)**2.0 * (np.exp(Theta_E2/T)/(np.exp(Theta_E2/T)-1)**2.0)  
    return Cp_Einstein2


# In[12]:


def LCEinstein_Cp(T, *param):
    a1 = param[0]
    a2 = param[1]
    Theta_E = param[2]
    Theta_E2 = param[3]
    Cp_LCE = []
    Cp_LCE = a1 *  Einstein_Cp(T,*param) + a2 * Einstein_Cp2(T,*param) 
                #  -----port1----------#       #------part2----------#    
    return Cp_LCE


# ## 2.9 Segmented with LCEEinstein

# In[13]:


def SRLCEinstein_Cp(T,*param):
    Cp_SRLC = []
    Parameter_LCEinstein = []
    Parameter_LCEinstein.append(param[0])
    Parameter_LCEinstein.append(param[1])
    Parameter_LCEinstein.append(param[2])
    Parameter_LCEinstein.append(param[3])
    Parameter_BCM = []
    Parameter_BCM.append(param[4])
    Parameter_BCM.append(param[5])
    Parameter_BCM.append(param[6])
    Parameter_BCM.append(param[7])
    Cp_LCEE = LCEinstein_Cp(T, *Parameter_LCEinstein) 
    Cp_LC_bc = bcm_Cp(T,*Parameter_BCM)
    for i in range (len(Cp_LCEE)):
        Cp_LC_EE = Cp_LCEE[i] + Cp_LC_bc[i]
        Cp_SRLC .append(Cp_LC_EE)
    return Cp_SRLC



# # 3. Nonlinear Least Squares Method

# ## 3.1 Goodness of fit criteria

# In[83]:


def AIC(logLik, nparm,k=2):
    return -2*logLik + k*(nparm + 1) 
def BIC(logLik, nobs,nparm,k=2):
    return -2*logLik + k*np.log(nobs)
def RSE(RSS,nobs, nparm,k=2):
    return sqrt(RSS/(-2*k-2-nobs))


# ## 3.2 Define 'name of regression function' and 'fitting parameters'

# In[84]:



new_df = df[df.Temp>5]
func = LCEinstein_Cp
parmNames = ['a1','a2','Theta_E','Theta_E2']
initialGuess=[0.1861546, 0.8140585, 88.74206, 237.8332]


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


# In[87]:


print ('Non-linear least squares')
print ('Model: ' + func.__name__)
print( '')
print(fit_df)
print()
print ('Residual Standard Error: % 5.4f' % RMSE)
print ('Df: %i' % dof)
print('AIC:', AIC(logLik, nparm))
print('BIC:', BIC(logLik, nobs,nparm))
print('RSE:', RSE(RSS, nobs,nparm))


# In[88]:



# # 4 Fitting Curve

# In[91]:
# T_plot = np.linspace(1,1000)
# CpRes_RW1995= RW1995_Cp(T_plot,popt[0], popt[1], popt[2])


# df= pd.read_csv("Mg_Cp.csv", sep=",")
# experiments = sorted(df.Ref.unique()) 
# colors = plt.cm.rainbow(np.linspace(0, 1, len(experiments)))
# for ref, col in zip(experiments, colors):
#     sub_df = df[df.Ref==ref]
#     plt.scatter(sub_df.Temp, sub_df.Cp, color=col, label=ref)
# plt.plot(T_plot, CpRes_RW1995,label="CpRes_RW1995")
# plt.ylim(0, df.Cp.max()*1.1)
# plt.xlim(0, df.Temp.max()*1.1)
# plt.xlabel("Temperature, in K")
# plt.ylabel("Heat capacity, in J/mol*K")
# plt.legend(ncol=3, fontsize='small', loc='best', bbox_to_anchor=(1, 1))
# plt.show()
