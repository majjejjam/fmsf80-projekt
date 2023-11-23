#%%Import
import numpy as np
import scipy.stats as stats
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

#Vår data av blyhalter
Pb = pd.read_csv("Pb.csv", encoding='utf-8')
Pb.Lan = Pb.Lan.astype('category')
Pb['Lan'] = Pb['Lan'].astype('category')

#Vi skapar en kolumn med gått tid från 1975
Pb['Year1975'] = Pb.Year - 1975

#Vi delar upp datan per län
Pb_B = Pb.query("Lan=='Blekinge län'")
Pb_S = Pb.query("Lan=='Södermanlands län'")

T = Pb_S['Year1975'].values
Y = Pb_S['Pb'].values

#%% Enkel regression, Linjär modell Södermanland

reg_lin=smf.ols(formula='Pb~Year1975',data=Pb_S).fit()
print(reg_lin.summary())

#Våra parametrar för den linjära modellen
alpha, beta = reg_lin.params

#Vår linjära modells residualer
epsilon_lin=reg_lin.resid

#Vi visar våra resulat från den linjära modellen
#plt.scatter(T, Y)
sns.set(style="whitegrid")
sns.scatterplot(x=T, y = Y)

#Vår linjära ekvation med våra estimerade parametrar
plt.axline((0, alpha), slope=beta)
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.title('Södermanland linjär')
plt.savefig('Grafer/SödermanlandLinjär.png')
plt.show()

#%% Enkel regression differential model Södermanland

reg_exp = smf.ols(formula='np.log(Pb)~Year1975',data=Pb_S).fit()
print(reg_exp.summary())

#Våra parametrar för den exponentiella modellen
C, k = np.exp(reg_exp.params[0]), reg_exp.params[1]

#Vår linjära modells residualer
t = np.linspace(0, 40, 200)

#Estimerade värden från vår exponentiella modell
y = C*np.exp(k*t)

#Exponentiell residual
epsilon_exp = []
for n in Y:
    epsilon_exp.append(np.abs(C*np.exp(k*n) - n))

#Vi plottar jämförelsen mellan våra mätvärden och estimerade värden från den exponentiella modellen
#plt.scatter(T, np.exp(Y_log))
sns.set(style="whitegrid")
sns.scatterplot(x= T, y =Y)
plt.plot(t, y)
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.title('Södermanland exponentiell')
plt.savefig('Grafer/SödermanlandExponentiell.png')
plt.show()

#%%Jämförelse mellan linjär och exponentiell modell
fig,axs=plt.subplots(1, 2, constrained_layout=True)
axs[0].set_title('Linjär')
axs[1].set_title('Exponentiell')

#Histogram över residualer från våra respektive modeller
sns.histplot(x=epsilon_lin,stat='density',kde=True,ax=axs[0])
sns.histplot(x=epsilon_exp,stat='density',kde=True,ax=axs[1])
plt.savefig('Grafer/Histogram.png')
plt.show()

#%% Multipel Regression Exponentiell 
Pb['Lan_I'] = [0 if Lan == 'Blekinge län' else 1 for Lan in Pb['Lan'].values]
#Regression på våra logaritmerade blyhalter som en funktion av tid och län 
res_mult_log = smf.ols(formula='np.log(Pb)~Lan_I + Year1975 + Lan_I:Year1975',data=Pb).fit()
print(res_mult_log.summary())

#Våra parametrar från vår multipel regression beta_0,,,beta_4=beta[0]...beta[4]
beta0,beta1,beta2,beta3 = res_mult_log.params
beta=[beta0,beta1,beta2,beta3]
#Våra tidsvärden mellan 1975 och 2015
t = np.linspace(0, 50, 200)

#Värdena på våra exponentiella funktioner för respektive län
y_exp_S = np.exp(beta0+beta1)*np.exp(t*(beta2))
y_exp_B = np.exp(beta0)*np.exp(t*beta2)

#Scatterplots med våra exponentiella modeller för respektive län
sns.set(style="whitegrid")
sns.scatterplot(x=Pb_S['Year1975'].values, y=Pb_S['Pb'].values, c='blue', label = "Södermanland - Mätvärden")
sns.scatterplot(x=Pb_B['Year1975'].values, y=Pb_B['Pb'].values, c='red', label = "Blekinge - Mätvärden")

#Våran mätdata för respektive län
plt.plot(t, y_exp_S, c='blue', label = "Södermanland-Exp modell")
plt.plot(t, y_exp_B, c='red', label = "Blekinge - Exp modell")

#Lägger till rätt titlar på axlar
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.legend()
plt.title("Exponentiella modeller")

#Sparar och visar grafen med våra exponentiella modeller
plt.savefig('Grafer/ExpModeller.png')
plt.show()

#%% Prediktion 2025 exp
Pb_0=pd.DataFrame({'Year1975' : [50, 50],'Lan_I' : [0, 1]})
Pred=res_mult_log.get_prediction(Pb_0).summary_frame(alpha=0.05)
print(str(Pred))
Pred_S_2025=[np.exp(Pred['mean_ci_lower'][1]),np.exp(Pred['mean'][1]),np.exp(Pred['mean_ci_upper'][1])]
Pred_B_2025=[np.exp(Pred['mean_ci_lower'][0]),np.exp(Pred['mean'][0]),np.exp(Pred['mean_ci_upper'][0])]

print('PREDIKTION 2025'+'[Undre,Mitten,Övre]')
print('Blekinge:'+str(Pred_B_2025) )
print('Södermanland:'+str(Pred_S_2025) )

#%%Prediktion 10mg/g (Sörmland & Blekinge)
y0=np.log(10)
cov=res_mult_log.cov_params()

x0_S=(y0-beta0-beta1)/(beta2+beta3)+1975
x0_B=(y0-beta0)/(beta2)+1975
beta=[beta0,beta1,beta2,beta3]
sim=stats.multivariate_normal.rvs(mean=beta,cov=cov,size=10000)
beta_u,beta_ö = [],[]
for x in range(4):
    param = np.quantile(sim[:, x], [0.025, 0.975])
    beta_u.append(param[0])
    beta_ö.append(param[1])

print(str(beta_u))
print(str(beta))
print(str(beta_ö))
#använder inte param[3] för den ger orimliga intervall(typ ökad blyhalt). Är ändå insignifikant

x0_S_ö = (y0 - beta_ö[0] - beta_ö[1]) / (beta_ö[2]) + 1975
x0_S_u = (y0 - beta_u[0] - beta_u[1]) / (beta_u[2]) + 1975
x0_B_ö = (y0 - beta_ö[0]) / (beta_ö[2]) + 1975
x0_B_u = (y0 - beta_u[0]) / (beta_u[2]) + 1975
pred_S_10mg=[x0_S_u,x0_S,x0_S_ö]
pred_B_10mg=[x0_B_u,x0_B,x0_B_ö]

print("PREDICTION year for 10mg [Undre, Mitten, Övre]")
print('Blekinge:'+ str(pred_B_10mg))
print('Södermanland:'+ str(pred_S_10mg))



#%% Prediktion plottad
t= np.linspace(0, 200, 200)
#Södermanland undre och övre intervall
y_exp_S_u=np.exp(beta_u[0]+beta_u[1])*np.exp(t*(beta_u[2]))
y_exp_S_ö=np.exp(beta_ö[0]+beta_ö[1])*np.exp(t*(beta_ö[2]))
#Blekinge undre och övre intervall
y_exp_B_u=np.exp(beta_u[0])*np.exp(t*beta_u[2])
y_exp_B_ö=np.exp(beta_ö[0])*np.exp(t*beta_ö[2])

##Grafer med både undre och övre konfidensinvervall för respektive län
sns.set(style="whitegrid")
fig,axs=plt.subplots(2, 1, constrained_layout=True)

#Våra tre respektive modeller för Södermanland
sns.lineplot(x=t, y=y_exp_S, color='blue',
                label='Södermanland', ax=axs[0])
sns.lineplot(x=t, y=y_exp_S_u, color='lightblue',
                label='Södermanland undre', ax=axs[0])
sns.lineplot(x=t, y=y_exp_S_ö, color='darkblue',
                label='Södermanland övre', ax=axs[0])

#Namnger våra axlar korrekt
axs[0].set_xlabel("Tid (år)")
axs[0].set_ylabel("Bly (mg/kg mossa)")
axs[0].legend()
axs[0].set_title("Södermanland")

#Våra tre respektive modeller för Blekinge
sns.lineplot(x=t, y=y_exp_B, color='red', label='Blekinge', ax=axs[1])
sns.lineplot(x=t, y=y_exp_B_u, color='lightcoral',
                label='Blekinge undre', ax=axs[1])
sns.lineplot(x=t, y=y_exp_B_ö, color='darkred',
                label='Blekinge övre', ax=axs[1])

#Namnger våra axlar korrekt
axs[1].set_xlabel("Tid (år)")
axs[1].set_ylabel("Bly (mg/kg mossa)")
axs[1].legend()
axs[1].set_title("Blekinge")

#Justerar och sparar grafen
plt.tight_layout()
plt.savefig('Grafer/ExpModellerIntervall.png')
plt.show()

#%%En tabell med våra resultat
pred_data = [Pred_S_2025,Pred_B_2025]
columns = ['Mitten','Undre','Övre']
pred_data_2 = [[np.round(Pred_S_2025)],
        [np.round(Pred_B_2025)]]

columns_2 = ['Undre Gräns','Väntevärde','Övre Gräns']

pred_res_2025 = pd.DataFrame(data=pred_data, index=['Södermanland','Blekinge']
                  ,columns=columns)