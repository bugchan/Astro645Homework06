# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:36:23 2019

@author: Sandra Bustamante
"""

import numpy as np
import matplotlib.pyplot as plt
import NumericIntegrations as NI
import SetupPlots as SP
import pandas as pd
import time
from scipy.special import erf
from scipy.integrate import simps
#import numpy.sqrt as sqrt
#import numpy.exp as exp
#import numpy.pi as pi

#%% Definitions

def dvdr(r,x,v,*args):
    #For this case t=r, x=psi, v=dpsi/dr,dvdt=d^2psi/dr^2
    d=(-4.*np.pi*G*rho(x,rho1,sigma)-(2.*v/r))
    return d

def rho(x,rho1=1,sigma=1):
    d=rho1*((np.exp(x/sigma**2.)*erf(np.sqrt(x)/sigma))
            -(np.sqrt(4.*x/(np.pi*sigma**2.))*(1.+(2*x/(3.*sigma**2)))))
    return d

#%%
start = time.time()

b=2.65
h=0.001
W0=np.arange(1.,8.75,.25)
rho1,G,sigma=1.,1.,1.

s=(len(W0),int(b/h))
#rtArray=np.zeros(len(W0))
#xtArray=np.zeros(len(W0))
#MArray=np.zeros(len(W0))
#phi_rt=np.zeros(len(W0))
#phi0=np.zeros(len(W0))
rho0=np.zeros(len(W0))
Etnorm=np.zeros(len(W0))
rsq=np.zeros(len(W0))
logr=np.zeros(len(W0))
rArray=np.zeros(s)
xArray=np.zeros(s)
vArray=np.zeros(s)


x0=W0*sigma**2 
v0=0.
rc=np.sqrt(9.*sigma**2/(4.*np.pi*G*rho(x0)))

for i in range(len(W0)):
    IV=[x0[i],v0]
    r,x,v=NI.RK4(dvdr,h,b,h,IV,dim=1)
    #r,x,v=NI.leapfrog2D(dvdr,h,b,h,IV,dim=1)
    rArray[i]=r
    xArray[i]=x
    vArray[i]=v
    psi=x[x>0]
    rx=r[x>0]
    rt=rx[-1]
    xt=psi[-1]
    Mrt=4.*np.pi*simps(rho(psi)*rx**2,x=rx)
    phi_rt=-G*Mrt/rt
    phi0=np.abs(phi_rt-x0[i])
    phi=phi_rt-psi
    Et=np.pi*simps(phi*rho(psi)*rx**2,x=rx)
    Etnorm[i]=Et/(G*Mrt**2/(rt))
    logr[i]=np.log10(rt/rc[i])
    rho0[i]=rho(x0[i])*4.*np.pi*rt**3/(3.*Mrt)
    rsq[i]=(simps(rho(psi)*rx**4,x=rx)/simps(rho(psi)*rx**2,x=rx))/rt**2

end = time.time()
print('Time to run: %0.2f'%(end - start))
for i in range(len(W0)):
    print('W0:%1.2f, Etnorm:%1.2f, logr:%1.2f, rho0:%1.2e,rsq:%1.2e'%(W0[i],Etnorm[i],logr[i],rho0[i],rsq[i]))

#print('x:',x)

#%%

#plt.plot(phi0,logr)
#plt.semilogx(r,xArray[-1])
#plt.semilogx(r,xArray[0])
#%% Save Data to csv file

#Create a dictionary with column title in latex and variables 
#can be a list or array.   
    
d = {'$W_0$': W0,
     '$E_t/(GM^2/r_t)$': Etnorm,
     '$\log_{10}(r_t/r_c)$': logr,
     '$\rho(0)/(3M/(4\pi r_t^3))$': rho0,
     '$<r^2>/r_t^2$': rsq }

#Create the table in tabular form
df = pd.DataFrame(d)

#Create the file.tex 
with open('KingModelTable.tex','w') as tf:
    tf.write(df.to_latex(float_format='%2.2e',
                         index=False,
                         escape=False))
    
#On latex, just do
#\begin{table}[]
#    \centering
#    \include{KingModel.tex}
#    \caption{Caption}
#    \label{tab:my_label}
#\end{table}
    


