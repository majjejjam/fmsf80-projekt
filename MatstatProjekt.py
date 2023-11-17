import numpy as np
import scipy.stats as stats
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

Pb = pd.read_csv("Documents/Python/MatStat Projekt/Pb.csv", encoding='utf-8')
Pb.Lan = Pb.Lan.astype('category')
Pb['Year1975'] = Pb.Year - 1975

#print( Pb.describe() )

Pb_S=Pb.loc[Pb['Lan'] == 'Södermanlands län']
Pb_B=Pb.loc[Pb['Lan']=='Blekinge län']
#print(Pb_S)
#print(Pb_B)

#Enkel regression, Linjär modell 
T=Pb_S['Year1975'].values
Y=Pb_S['Pb'].values

t_reg=sm.add_constant(T)
res_lin=sm.OLS(Y,t_reg).fit()
print(res_lin.summary())

a,b=res_lin.params
plt.scatter(T,Y)
plt.axline((0,a),slope=b)
plt.show()

#Enkel regression differential model
Y_log=np.log(Y)
res_log=sm.OLS(Y_log,t_reg).fit()
print(res_log.summary())


C,k=np.exp(res_log.params[0]),res_log.params[1]

plt.scatter(T,Y)
t_line=np.linspace(0,40,200)
y_exp=C*np.exp(k*t_line)
plt.plot(t_line,y_exp)
plt.show()

print(C*np.exp(k*50))
print(a+b*50)

#Multipel Regression osv nånting.....

Platser = Pb['Lan'].values
P = [0 if plats == 'Blekinge län' else 1 for plats in Platser]
T=Pb['Year1975'].values
X=list(zip(T,P))
Y=Pb['Pb'].values

x_reg=sm.add_constant(X)
res_mult=sm.OLS(Y,x_reg).fit()
print(res_mult.summary())
#plt.scatter(T,Y)
#plt.show()

