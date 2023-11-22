# -*- coding: utf-8 -*-
"""
Enkel linjär regression för avtagande blyhalt över tid.

@author: Johan and Miró
"""

# %% Import modules
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# %% Data
Pb = pd.read_csv("Pb.csv", encoding='utf-8')
Pb.Lan = Pb.Lan.astype('category')

# Vi skapar en kolumn med gått tid från 1975
Pb['Year1975'] = Pb.Year - 1975

# Vi delar upp datan per län
Pb_B = Pb.query("Lan=='Blekinge län'")
Pb_S = Pb.query("Lan=='Södermanlands län'")

# %% plot data
plt.figure()
sns.scatterplot(Pb, x='Year', y='Pb', hue='Lan')
plt.show()

# %% enkel linjär regression
# beräkna konstanter
n = Pb_S.shape[0]
x_bar = np.mean(Pb_S.Year1975)
y_bar = np.mean(Pb_S.Pb)
Sxx = sum((Pb_S.Year1975-x_bar)**2)
Syy = sum((Pb_S.Pb-y_bar)**2)
Sxy = sum((Pb_S.Year1975-x_bar)*(Pb_S.Pb-y_bar))

# skattningar
beta = Sxy/Sxx
alpha = y_bar - beta*x_bar
Q0 = Syy - Sxy**2 / Sxx
s = np.sqrt(Q0/(n-2))

print('Skattning av linjen y = ' + format(alpha, '.4f') + ' + ' +
      format(beta, '.4f') + 'x')
print('Med sigma ' + format(s, '.3f'))

# %% På matris form
X = np.array([np.ones(n), Pb_S.Year1975]).T
Beta = np.linalg.lstsq(X, Pb_S.Pb, rcond=None)[0]
# eller np.linalg.solve(X.T.dot(X), X.T.dot(Cu.y))

# residualer (utan np.array blir det bara n series)
res = np.array(Pb_S.Pb-X@Beta)

# skattning av s
s = np.sqrt(sum(res**2) / (Pb_S.shape[0]-Beta.shape[0]))

print('Skattning av linjen y = ' + format(Beta[0], '.4f') + ' + ' +
      format(Beta[1], '.4f') + 'x')
print('Med sigma ' + format(s, '.3f'))

# %% plot data och linjen
sns.scatterplot(Pb_S, x='Year1975', y='Pb')
plt.axline((0, alpha), slope=beta, color='r')
plt.title('Regressionslinje anpassad till data')
plt.ylabel('Absorption')
plt.xlabel('Kopparkoncentration')
plt.show()

# %% Intervall för alpha, beta
# först beräknar vi kvantilerna
t_alpha = stats.t.ppf(0.975, n-2)
# define a 2-element array with - and + quantile
t_alpha_pm = np.array([-1, 1]) * t_alpha

I_beta = beta + t_alpha_pm * s/np.sqrt(Sxx)
I_alpha = alpha + t_alpha_pm * s * np.sqrt(1/n + x_bar**2/Sxx)

# eller på matris form
XtX = np.linalg.inv(X.T @ X)
D_Beta = s * np.sqrt(np.diag(XtX))

f = X.shape[0]-X.shape[1]
t_alpha = stats.t.ppf(0.975, f)
t_alpha_pm = np.array([-1, 1]) * t_alpha

I_Beta = np.column_stack((Beta, Beta)) + \
    t_alpha * np.column_stack((-D_Beta.T, D_Beta.T))

# enkel regression med statsmodel
res_ols = smf.ols(formula='Pb ~ Year1975', data=Pb_S)
# resultat
print(res_ols.fit().summary())
# parametrar och intervall
print(res_ols.fit().params)
print(res_ols.fit().conf_int())

# %% konfidens och prediktions intervall
x0 = 50  # år 2025

# kvantiler har beräknats ovan

# simple linear regression
mu0 = alpha + beta*x0
D_mu = s * np.sqrt(1/n + (x0-x_bar)**2 / Sxx)
I_mu0 = mu0 + t_alpha_pm * D_mu

D_y = s * np.sqrt(1 + 1/n + (x0-x_bar)**2 / Sxx)
I_y0 = alpha + beta*x0 + t_alpha_pm * D_y

# on matrix form
X0 = np.array([1, x0])

I_mu0 = X0@Beta + t_alpha_pm * s * np.sqrt(X0 @ np.linalg.solve(X.T@X, X0))
I_y0 = X0@Beta + t_alpha_pm * s * np.sqrt(1 + X0 @ np.linalg.solve(X.T@X, X0))

# %% plot confidence bounds
# kvantiler har beräknats ovan
# create a vector of values
t = np.linspace(-25, 225, 10**3)
# and a matrix
T = np.column_stack((np.ones(t.shape[0]), t))
# predictions and variances
mu = T@Beta
V = np.sum((T @ np.linalg.inv(X.T@X)) * T, axis=1)

# plot data
sns.scatterplot(Pb_S, x='Year1975', y='Pb', color='k')
plt.plot(t, mu, color='b')
plt.plot(t, mu+t_alpha*s*np.sqrt(V), color='g')
plt.plot(t, mu-t_alpha*s*np.sqrt(V), color='g')
plt.plot(t, mu+t_alpha*s*np.sqrt(V+1), color='r')
plt.plot(t, mu-t_alpha*s*np.sqrt(V+1), color='r')

plt.plot([-25, x0], I_mu0[[0, 0]], color='grey')
plt.plot([-25, x0], I_mu0[[1, 1]], color='grey')
plt.plot([-25, x0], I_y0[[0, 0]], color='grey')
plt.plot([-25, x0], I_y0[[1, 1]], color='grey')
plt.plot([x0, x0], [-0.11, I_y0[1]], color='grey')

plt.title('Konfidensintervall för mu(' + format(x0, '.2f') +
          ') och prediktionsintervall')
plt.ylabel('Absorption')
plt.xlabel('Kopparkoncentration')
plt.show()

# %% kalibrationintervalls
y0 = 10  # under 10 mg
x0_star = (y0-alpha)/beta

# simple linear regression
D_x0 = s/abs(beta) * np.sqrt(1 + 1/n + (y0-y_bar)**2 / (beta**2 * Sxx))
I_x0 = x0_star + t_alpha_pm * D_x0

# %% plot calibration
# mu och V har beräknats ovan för konfidens intervallet

# plot data
sns.scatterplot(Pb_S, x='Year1975', y='Pb', color='k')
plt.plot(t, mu, color='b')
plt.plot(t, mu+t_alpha*s*np.sqrt(V), color='g')
plt.plot(t, mu-t_alpha*s*np.sqrt(V), color='g')
plt.plot(t, mu+t_alpha*s*np.sqrt(V+1), color='r')
plt.plot(t, mu-t_alpha*s*np.sqrt(V+1), color='r')

plt.plot([-25, I_x0[1]], [y0, y0], color='grey')
plt.plot(I_x0[[0, 0]], [-0.11, y0], color='grey')
plt.plot(I_x0[[1, 1]], [-0.11, y0], color='grey')
plt.plot([x0_star, x0_star], [-0.11, y0], color='grey')

plt.title('Kalibreringsintervall för y_0=' + format(y0, '.2f'))
plt.ylabel('Absorption')
plt.xlabel('Kopparkoncentration')
plt.show()

print("Konfidensintervall för blyhalt 2025: ", I_mu0)
print("Konfidensintervall för årtal då blyhalt understiger 10 mg: ", I_x0 + 1975)
