# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:36:23 2019

@author: Sandra Bustamante

"""

# import numpy as np
import matplotlib.pyplot as plt
import NumericIntegrations as NI
import SetupPlots as SP
import pandas as pd
import time
from scipy.special import erf
from scipy.integrate import simps
from numpy import sqrt, exp, pi, abs, log10
from numpy import zeros, arange, empty
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# %% Definitions


def dvdr(r, x, v, *args):
    """Define function to be integrated.

    Parameters
    ----------
    r : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    v : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    """
    # For this case t=r, x=psi, v=dpsi/dr,dvdt=d^2psi/dr^2
    d = (-4.*pi*G*rho(x, rho1, sigma)-(2.*v/r))
    return d


def rho(x, rho1=1, sigma=1):
    """Calculate the King density.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    rho1 : TYPE, optional
        DESCRIPTION. The default is 1.
    sigma : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    d : TYPE
        DESCRIPTION.

    """
    d = rho1*((exp(x/sigma**2.)*erf(sqrt(x)/sigma))
              - (sqrt(4.*x/(pi*sigma**2.))*(1.+(2*x/(3.*sigma**2)))))
    return d


# %%
start = time.time()

b = 2.7
h = 0.0001
W0 = arange(1., 8.75, .25)
rho1, G, sigma = 1., 1., 1.

s = (len(W0), int(b/h))

rho0 = zeros(len(W0))
Etnorm = zeros(len(W0))
rsq = zeros(len(W0))
logr = zeros(len(W0))

rxArray = empty(s)
psiArray = empty(s)
rhoArray = empty(s)

x0 = W0*sigma**2
v0 = 0.
rc = sqrt(9.*sigma**2/(4.*pi*G*rho(x0)))

for i in range(len(W0)):
    IV = [x0[i], v0]
    # r, x, v = NI.RK4(dvdr, h, b, h, IV, dim=1)
    r, x, v = NI.leapfrog2D(dvdr, h, b, h, IV, dim=1)
    psi = x[x > 0]
    psiArray[i] = x
    rx = r[x > 0]
    rxArray[i] = r
    rt = rx[-1]
    xt = psi[-1]
    Mrt = 4.*pi*simps(rho(psi)*rx**2, x=rx)
    phi_rt = -G*Mrt/rt
    phi0 = abs(phi_rt-x0[i])
    phi = phi_rt-psi
    Et = pi*simps(phi*rho(psi)*rx**2, x=rx)
    Etnorm[i] = Et/(G*Mrt**2/(rt))
    logr[i] = log10(rt/rc[i])
    rho0[i] = rho(x0[i])*4.*pi*rt**3/(3.*Mrt)
    rhoArray[i] = rho(x)
    rsq[i] = (simps(rho(psi)*rx**4, x=rx)/simps(rho(psi)*rx**2, x=rx))/rt**2

end = time.time()
print('Time to run: %0.2f' % (end - start))

# %% Plot

width, height = SP.setupPlot(singleColumn=False)
fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0, width_ratios=[10., .5])
fig.set_size_inches((width, width))

# setup the normalization and the colormap
normalize = mcolors.Normalize(vmin=W0.min(), vmax=W0.max())
colormap = cm.plasma

ax1 = fig.add_subplot(gs[0, :1])
ax2 = fig.add_subplot(gs[1, :1])
for i in range(len(W0)):
    ax1.semilogx(rxArray[i][rhoArray[i] > 0],
                 log10(rhoArray[i][rhoArray[i] > 0]/rho0[i]),
                 color=colormap(normalize(W0[i])))
    ax2.semilogx(rxArray[i][rhoArray[i] > 0],
                 psiArray[i][rhoArray[i] > 0],
                 color=colormap(normalize(W0[i])))

ax1.set_xlabel(r'$r$')
ax1.set_ylabel(r'$\rho(r)/\rho(0)$')
ax1.set_ylim((-6, 0))
ax1.set_xticks([])
ax1.grid()

ax2.set_ylabel(r'$\psi(r)$')
ax2.grid()

# setup the colorbar
scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(W0)
axc = fig.add_subplot(gs[:2, 1])
fig.colorbar(scalarmappaple, cax=axc, label='$W_0$')

fig.tight_layout()
fig.savefig('KingModelPlots.pdf')

# %% Save Data to latex table.

# Create a dictionary with column title in latex and variables
# can be a list or array.

d = {r'$W_0$': W0,
     r'$E_t/(GM^2/r_t)$': Etnorm,
     r'$\log_{10}(r_t/r_c)$': logr,
     r'$\rho(0)/(3M/(4\pi r_t^3))$': rho0,
     r'$<r^2>/r_t^2$': rsq}

# Create the table in tabular form
df = pd.DataFrame(d)

print(df)

# Create the file.tex
with open('KingModelTable.tex', 'w') as tf:
    tf.write(df.to_latex(float_format='%2.2e',
                         index=False,
                         escape=False))

# In latex, just do:
# \begin{table}[]
#    \centering
#    \input{KingModel.tex}
#    \caption{Caption}
#    \label{tab:my_label}
# \end{table}
