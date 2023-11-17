import numpy as np
import scipy.stats as stats
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

Pb = pd.read_csv("Pb.csv", encoding='utf-8')
Pb.Lan = Pb.Lan.astype('category')
Pb['Year1975'] = Pb.Year - 1975

#print( Pb.describe() )

Pb_S=Pb.loc[Pb['Lan'] == 'Södermanlands län']
Pb_B=Pb.loc[Pb['Lan']=='Blekinge län']
#print(Pb_S)
#print(Pb_B)


#Enkel regression, Linjär modell
def linReg(Pb,Län):
    Pb_Lan=Pb.loc[Pb['Lan'] == Län]
    T=Pb_Lan['Year1975'].values
    Y=Pb_Lan['Pb'].values

    t_reg=sm.add_constant(T)
    res_lin=sm.OLS(Y,t_reg).fit()
    print(res_lin.summary())

    a,b=res_lin.params
    plt.scatter(T,Y)
    plt.axline((0,a),slope=b)
    plt.xlabel("Year")
    plt.ylabel("Pb")
    plt.title(Län)
    plt.show()

linReg(Pb,'Blekinge län')
linReg(Pb,'Södermanlands län')

#Enkel regression differential model
def expReg(Pb,Län):
    Pb_Lan=Pb.loc[Pb['Lan'] == Län]

    T=Pb_Lan['Year1975'].values
    t_reg=sm.add_constant(T)
    Y=np.log(Pb_Lan['Pb'].values)

    res=sm.OLS(Y,t_reg).fit()
    print(res.summary())

    C,k=np.exp(res.params[0]),res.params[1]
    t=np.linspace(0,40,200)
    y=C*np.exp(k*t)
    
    plt.scatter(T,np.exp(Y))
    plt.plot(t,y)
    plt.xlabel("Year")
    plt.ylabel("Pb")
    plt.title(Län)
    plt.show()

expReg(Pb,'Blekinge län')


#Multipel Regression
def multReg(Pb):
    Pb_S=Pb.loc[Pb['Lan'] == 'Södermanlands län']
    Pb_B=Pb.loc[Pb['Lan']=='Blekinge län']

    Y_log=np.log(Pb['Pb'].values)
    Platser = Pb['Lan'].values
    P = [0 if plats == 'Blekinge län' else 1 for plats in Platser]
    T=Pb['Year1975'].values
    X=list(zip(T,P))


    x_reg=sm.add_constant(X)
    res_mult_log=sm.OLS(Y_log,x_reg).fit()
    print(res_mult_log.summary())
    C,k1,k2=np.exp(res_mult_log.params[0]),res_mult_log.params[1],res_mult_log.params[2]
    
    t=np.linspace(0,40,200)
    y_exp_S=C*np.exp(t*k1)*np.exp(k2)
    y_exp_B=C*np.exp(t*k1)

    plt.scatter(Pb_S['Year1975'].values,Pb_S['Pb'].values,c='blue')
    plt.scatter(Pb_B['Year1975'].values,Pb_B['Pb'].values,c='red')
    plt.plot(t,y_exp_S,c='blue')
    plt.plot(t,y_exp_B,c='red')
    plt.show()


#Multipel regression exponentiell men där vi tar bort år 20 (chernobyl)
Pb_ny=Pb[Pb['Year1975']!=20]
multReg(Pb_ny)
