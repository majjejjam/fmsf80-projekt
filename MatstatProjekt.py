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

Y_log = np.log(Y)
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
sns.scatterplot(x= T, y =np.exp(Y_log))
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
Pb['log_Pb']=np.log(Pb['Pb'])
#Regression på våra logaritmerade blyhalter som en funktion av tid och län 
res_mult_log = smf.ols(formula='log_Pb~Lan_I + Year1975 + Lan_I:Year1975',data=Pb).fit()
print(res_mult_log.summary())

#Våra parametrar från vår multipel regression beta_0,,,beta_4=beta[0]...beta[4]
beta_exp = [res_mult_log.params['Intercept'], res_mult_log.params['Lan_I'], res_mult_log.params['Year1975'], res_mult_log.params['Lan_I:Year1975']]

#Våra tidsvärden mellan 1975 och 2015
t = np.linspace(0, 40, 200)

#Värdena på våra exponentiella funktioner för respektive län
y_exp_S = np.exp(beta_exp[0]+beta_exp[1])*np.exp(t*(beta_exp[2]+beta_exp[3]))
y_exp_B = np.exp(beta_exp[0])*np.exp(t*beta_exp[2])


#plt.scatter(Pb_S['Year1975'].values, Pb_S['Pb'].values, c='blue')
#plt.scatter(Pb_B['Year1975'].values, Pb_B['Pb'].values, c='red')

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

#%% Lineär modell Multipel regression
res_mult_lin = smf.ols(formula='Pb ~ Lan_I + Year1975 + Lan_I:Year1975', data=Pb).fit()
print(res_mult_lin.summary())
beta_lin=[res_mult_lin.params['Intercept'], res_mult_lin.params['Lan_I'], res_mult_lin.params['Year1975'], res_mult_lin.params['Lan_I:Year1975']]
#%% Prediktion 2025 exp
Pb_0=pd.DataFrame({'Year1975' : [50, 50],'Lan_I' : [0, 1]})
Pred=res_mult_log.get_prediction(Pb_0).summary_frame(alpha=0.05)

Pred_S_2025=[np.exp(Pred['mean'][1]),np.exp(Pred['mean_ci_lower'][1]),np.exp(Pred['mean_ci_upper'][1])]##ska vi ha mean/obs här??
Pred_B_2025=[np.exp(Pred['mean'][0]),np.exp(Pred['mean_ci_lower'][0]),np.exp(Pred['mean_ci_upper'][0])]
print('PREDIKTION 2025'+'[Mitten,Undre,Övre]')
print('Blekinge:'+str(Pred_B_2025) )
print('Södermanland:'+str(Pred_S_2025) )

#%%Nytt försök Prediktion 10mg/g (Sörmland & Blekinge)
def pred_10mg(reg_mult,beta,y0):
    cov=reg_mult.cov_params()

    x0_S=(y0-beta[0]-beta[1])/(beta[2]+beta[3])+1975
    x0_B=(y0-beta[0])/(beta[2])+1975
    sim=stats.multivariate_normal.rvs(mean=beta,cov=cov,size=10000)
    params = []
    for x in range(4):
    # Adjusted quantile values to get the 2.5th and 97.5th percentiles
        param = np.quantile(sim[:, x], [0.025, 0.975])
        params.append(param)
    print(str(params))
 #anv inte param[3] för den ger orimliga intervall. Är ändå insignifikant
# Adjusted indices when calculating upper and lower bounds
    x0_S_Ö = (y0 - params[0][1] - params[1][1]) / (params[2][1]) + 1975
    x0_S_U = (y0 - params[0][0] - params[1][0]) / (params[2][0]) + 1975

    x0_B_Ö = (y0 - params[0][1]) / (params[2][1]) + 1975
    x0_B_U = (y0 - params[0][0]) / (params[2][0]) + 1975

    print("PREDICTION year for 10mg [Undre, Mean, Övre]")
    print('Blekinge:', str(x0_B_U) + ', ' + str(x0_B) + ', ' + str(x0_B_Ö))
    print('Södermanland:', str(x0_S_U) + ', ' + str(x0_S) + ', ' + str(x0_S_Ö))


pred_10mg(res_mult_log,beta_exp,np.log(10))
pred_10mg(res_mult_lin,beta_lin,10)


#%%Prediktion när under 10mg/g mossa (Sörmland) (Gaussapproximation)

y_0=10
y_mean = np.mean(np.log(Pb_S['Pb'].values))

t=stats.t.ppf(0.975,n-2)

x_0_approx=(np.log(y_0)-np.log(C)-k2*1)/k1
cov=res_mult_log.cov_params()
s=np.sqrt((cov['Year1975'][1]))
print(str(s))
x_years1975=[0,5,10,15,20,25,30,35,40]
n=len(Pb_S['Pb'].values)
x_mean=(np.mean(Pb_S['Year1975']))
s_yy=sum((y-y_mean)**2 for y in np.log(Pb_S['Pb'].values))
s_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(Pb_S['Year1975'], np.log(Pb_S['Pb'].values)))
s_xx=sum((x-x_mean)**2 for x in Pb_S['Year1975'])
#s=np.sqrt((s_yy-(s_xy**2)/s_xx)/(n-2))
print(str(s))
n=len(x_years1975)

width_interval=t*(s/np.abs(k1))*np.sqrt(1+1/n+(x_0_approx-x_mean)**2)/((s_xx))
print(str(width_interval))
x_0_final=1975+x_0_approx

pred=[x_0_final+width_interval,x_0_final,x_0_final-width_interval]
print('Prediktion när mg/g mossa är under 10')
print('[över, medel, under]')
print("Södermanland: "+str(pred))
#%% Prediktion 10mg (Blekinge)
y_0=10
y_mean = np.mean(np.log(Pb_B['Pb'].values))
n=len(Pb_B['Pb'])
t=stats.t.ppf(0.975,n-2)

x_0_approx=(np.log(y_0)-np.log(C))/k1
cov=res_mult_log.cov_params()
s=np.sqrt((cov['Year1975'][1]))
x_mean=np.mean(Pb_B['Year1975'])
s_xx=sum((x-x_mean)**2 for x in Pb_B['Year1975'].values)
s_xx=1

width_interval=t*(s/np.abs(k1))*np.sqrt(1+1/n+((np.log(y_0)-y_mean)**2)/((k1**2)*s_xx))
x_0_final=1975+x_0_approx

pred=[x_0_final+width_interval,x_0_final,x_0_final-width_interval]
print('Prediktion när mg/g mossa är under 10')
print('[över, medel, under]')
print("Blekinge: "+str(pred))
#%% Prediktion plottad
#Våra konfidensintervall för vår multipel regression modell 
intervals=res_mult_log.conf_int()
C_h, k1_h, k2_h = np.exp(intervals[1]['Intercept']), intervals[1]['Year1975'], intervals[1]['Lan_I']
C_l, k1_l, k2_l = np.exp(intervals[0]['Intercept']), intervals[0]['Year1975'], intervals[0]['Lan_I']

#Våran exponentiella modell
def exp(C,k1,k2,t):
    return C*np.exp(t*k1)*np.exp(k2)

t = np.linspace(0, 40, 200)

#Södermanland undre och övre intervall
y_exp_S_undre=exp(C_l,k1_l,k2_l,t)
y_exp_S = exp(C,k1,k2,t)
y_exp_S_övre=exp(C_h,k1_h,k2_h,t)

#Blekinge undre och övre intervall
y_exp_B_undre=exp(C_l,k1_l,0,t)
y_exp_B = exp(C,k1,0,t)
y_exp_B_övre=exp(C_h,k1_h,0,t)

##Grafer med både undre och övre konfidensinvervall för respektive län
sns.set(style="whitegrid")
fig,axs=plt.subplots(2, 1, constrained_layout=True)

#Våra tre respektive modeller för Södermanland
sns.lineplot(x=t, y=y_exp_S, color='blue',
                label='Södermanland', ax=axs[0])
sns.lineplot(x=t, y=y_exp_S_undre, color='lightblue',
                label='Södermanland undre', ax=axs[0])
sns.lineplot(x=t, y=y_exp_S_övre, color='darkblue',
                label='Södermanland övre', ax=axs[0])

#Namnger våra axlar korrekt
axs[0].set_xlabel("Tid (år)")
axs[0].set_ylabel("Bly (mg/kg mossa)")
axs[0].legend()
axs[0].set_title("Södermanland")

#Våra tre respektive modeller för Blekinge
sns.lineplot(x=t, y=y_exp_B, color='red', label='Blekinge', ax=axs[1])
sns.lineplot(x=t, y=y_exp_B_undre, color='lightcoral',
                label='Blekinge undre', ax=axs[1])
sns.lineplot(x=t, y=y_exp_B_övre, color='darkred',
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

##Våran prediktion av blyvärdet 2025 för respektive län
#Södermanland
pred_exp_S_undre=round(exp(C_l,k1_l,k2_l,50),2)#dessa  stämmer inte överens med tidigare predictions
pred_exp_S = round(exp(C,k1,k2,50),2)
pred_exp_S_övre=round(exp(C_h,k1_h,k2_h,50),2)

#Blekinge
pred_exp_B_undre=round(exp(C_l,k1_l,0,50),2)
pred_exp_B = round(exp(C,k1,0,50),2)
pred_exp_B_övre= round(exp(C_h,k1_h,0,50),2)

#Presentation av resultat för värden år 2025
print('PREDIKTION 2025[Övre, medel,Undre]')
print('Blekinge: '+str(pred_exp_B_övre)+', '+str(pred_exp_B)+', '+str(pred_exp_B_undre))
print('Södermanland: '+str(pred_exp_S_övre)+', '+str(pred_exp_S)+', '+str(pred_exp_S_undre))

#En tabell med våra resultat
pred_data = [Pred_S_2025,Pred_B_2025]
columns = ['Mitten','Undre','Övre']
pred_data_2 = [[pred_exp_S_undre,pred_exp_S,pred_exp_S_övre],
        [pred_exp_B_undre,pred_exp_B,pred_exp_B_övre]]

columns_2 = ['Undre Gräns','Väntevärde','Övre Gräns']

pred_res_2025 = pd.DataFrame(data=pred_data, index=['Södermanland','Blekinge']
                  ,columns=columns)



#%%Latex-tabeller för våra regressionsresultat
latex_table_lin = reg_lin.summary().as_latex()
latex_table_exp = reg_exp.summary().as_latex()
latex_table_mult_log = res_mult_log.summary().as_latex()
latex_prediktioner_2025 = pred_res_2025.to_latex()

with open('LatexTabeller/regression_table_lin.tex', 'w') as f:
    f.write(latex_table_lin)

with open('LatexTabeller/regression_table_exp.tex', 'w') as f:
    f.write(latex_table_exp)

with open('LatexTabeller/regression_table_mult_log.tex', 'w') as f:
    f.write(latex_table_mult_log)

with open('LatexTabeller/prediktioner_2025.tex', 'w') as f:
    f.write(latex_prediktioner_2025)


## Multipel regression exponentiell men där vi flyttar bak alla år efter 1995 med 20 år (chernobyl)
#Pb[Pb['Year1975'] != 20]
Pb_ny = Pb.copy()
Pb_ny.loc[Pb_ny['Year1975'] >= 20, 'Year1975'] -= 20
Pb_S_ny = Pb_ny.loc[Pb['Lan'] == 'Södermanlands län']
Pb_B_ny = Pb_ny.loc[Pb['Lan'] == 'Blekinge län']

Y_log = np.log(Pb_ny['Pb'].values)

#Vi skapar en dummy-variabel för våra län där Blekinge ges en 0 och Södermanland ges en 1a
Platser = Pb_ny['Lan'].values
P = [0 if plats == 'Blekinge län' else 1 for plats in Platser]

#Vi kombinerar dessa med våra tidsvärden som tid från år 1975
T = Pb_ny['Year1975'].values
X = list(zip(T, P))

#Regression på våra logaritmerade blyhalter som en funktion av tid och län
x_reg = sm.add_constant(X)
res_mult_log = sm.OLS(Y_log, x_reg).fit()
print(res_mult_log.summary())

#Våra parametrar från vår multipel regression
C, k1, k2 = np.exp(res_mult_log.params[0]), res_mult_log.params[1], res_mult_log.params[2]

#Våra tidsvärden mellan 1975 och 2015
t = np.linspace(0, 40, 200)

#Värdena på våra exponentiella funktioner för respektive län
y_exp_S = C*np.exp(t*k1)*np.exp(k2)
y_exp_B = C*np.exp(t*k1)

#plt.scatter(Pb_S['Year1975'].values, Pb_S['Pb'].values, c='blue')
#plt.scatter(Pb_B['Year1975'].values, Pb_B['Pb'].values, c='red')

#Scatterplots med våra exponentiella modeller för respektive län
sns.scatterplot(x=Pb_S_ny['Year1975'].values, y=Pb_S_ny['Pb'].values, c='blue', label = "Södermanland - Exponentiell modell")
sns.scatterplot(x=Pb_B_ny['Year1975'].values, y=Pb_B_ny['Pb'].values, c='red', label = "Blekinge - Exponentiell modell")

#Våran mätdata för respektive län
plt.plot(t, y_exp_S, c='blue', label = "Södermanland - Mätvärden")
plt.plot(t, y_exp_B, c='red', label = "Blekinge - Mätvärden")

#Lägger till rätt titlar på axlar
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.legend()
plt.title("Exponentiella modeller")

#Sparar och visar grafen med våra exponentiella modeller
plt.savefig('GraferAvvikande/ExpModellerAvvikande.png')
plt.show()


## Prediktion med intervall
#Våra konfidensintervall för vår multipel regression modell
intervals=res_mult_log.conf_int()
C_h,k1_h,k2_h=np.exp(intervals[0][1]),intervals[1][1],intervals[2][1]
C_l,k1_l,k2_l=np.exp(intervals[0][0]),intervals[1][0],intervals[2][0]

#Våran exponentiellla modell
def exp(C,k1,k2,t):
    return C*np.exp(t*k1)*np.exp(k2)

#Södermanland undre och övre intervall
y_exp_S_undre=exp(C_l,k1_l,k2_l,t)
y_exp_S = exp(C,k1,k2,t)
y_exp_S_övre=exp(C_h,k1_h,k2_h,t)

#Blekinge undre och övre intervall
y_exp_B_undre=exp(C_l,k1_l,0,t)
y_exp_B = exp(C,k1,0,t)
y_exp_B_övre=exp(C_h,k1_h,0,t)

##Grafer med både undre och övre konfidensinvervall för respektive län
sns.set(style="whitegrid")
fig,axes=plt.subplots(2, 1, constrained_layout=True)

#Våra tre respektive modeller för Södermanland
sns.lineplot(x=t, y=y_exp_S, color='blue',
                label='Södermanland', ax=axes[0])
sns.lineplot(x=t, y=y_exp_S_undre, color='lightblue',
                label='Södermanland undre', ax=axes[0])
sns.lineplot(x=t, y=y_exp_S_övre, color='darkblue',
                label='Södermanland övre', ax=axes[0])

#Namnger våra axlar korrekt
axes[0].set_xlabel("Tid (år)")
axes[0].set_ylabel("Bly (mg/kg mossa)")
axes[0].legend()
axes[0].set_title("Södermanland")

#Våra tre respektive modeller för Blekinge
sns.lineplot(x=t, y=y_exp_B, color='red', label='Blekinge', ax=axes[1])
sns.lineplot(x=t, y=y_exp_B_undre, color='lightcoral',
                label='Blekinge undre', ax=axes[1])
sns.lineplot(x=t, y=y_exp_B_övre, color='darkred',
                label='Blekinge övre', ax=axes[1])

#Namnger våra axlar korrekt
axes[1].set_xlabel("Tid (år)")
axes[1].set_ylabel("Bly (mg/kg mossa)")
axes[1].legend()
axes[1].set_title("Blekinge")

#Justerar och sparar grafen
plt.tight_layout()
plt.savefig('GraferAvvikande/ExpModellerIntervallAvvikande.png')
plt.show()

##Våran prediktion av blyvärdet 2025 för respektive län
#Södermanland
pred_exp_S_undre=round(exp(C_l,k1_l,k2_l,50),2)
pred_exp_S = round(exp(C,k1,k2,50),2)
pred_exp_S_övre=round(exp(C_h,k1_h,k2_h,50),2)

#Blekinge
pred_exp_B_undre=round(exp(C_l,k1_l,0,50),2)
pred_exp_B = round(exp(C,k1,0,50),2)
pred_exp_B_övre= round(exp(C_h,k1_h,0,50),2)

#Presentation av resultat för värden år 2025
print('PREDIKTION 2025[Övre, medel,Undre]')
print('Blekinge: '+str(pred_exp_B_övre)+', '+str(pred_exp_B)+', '+str(pred_exp_B_undre))
print('Södermanland: '+str(pred_exp_S_övre)+', '+str(pred_exp_S)+', '+str(pred_exp_S_undre))

#En tabell med våra resultat
data = [[pred_exp_S_undre,pred_exp_S,pred_exp_S_övre],
        [pred_exp_B_undre,pred_exp_B,pred_exp_B_övre]]

pred_res_2025_avvikande = pd.DataFrame(data=data, index=['Södermanland','Blekinge']
                  ,columns=['Undre Gräns','Väntevärde','Övre Gräns'])



#Kommenterad gammal kod
#C, k1, k2 = np.exp(
#    res_mult_log.params[0]), res_mult_log.params[1], res_mult_log.params[2]
#C_se, k1_se, k2_se = np.exp(
#    res_mult_log.bse[0]), res_mult_log.bse[1], res_mult_log.bse[2]
#t = np.linspace(0, 100, 200)

#df = pd.DataFrame(index=t)
#df['Södermanland'] = y_exp_S
#df['Blekinge'] = y_exp_B

#df['Beta S'] = k1
#df['Beta B'] = k2
#df['Beta S - stdev'] = k1_se
#df['Beta B - stdev'] = k2_se
#df['Beta S - undre'] = k1 - 2*k1_se
#df['Beta B - undre'] = k2 - 2*k2_se
#df['Beta S - övre'] = k1 + 2*k1_se
#df['Beta B - övre'] = k2 + 2*k2_se

#y_exp_S_undre = C*np.exp(t*df['Beta S - undre']) * \np.exp(df['Beta B - undre'])
#y_exp_B_undre = C*np.exp(t*df['Beta S - undre'])
#y_exp_S_övre = C*np.exp(t*df['Beta S - övre'])*np.exp(df['Beta B - övre'])
#y_exp_B_övre = C*np.exp(t*df['Beta S - övre'])

#df['Södermanland - Undre'] = y_exp_S_undre
#df['Blekinge - Undre'] = y_exp_B_undre
#df['Södermanland - Övre'] = y_exp_S_övre
#df['Blekinge - Övre'] = y_exp_B_övre

Pb_0 = pd.DataFrame({'Year1975' : [50],'Lan' : [1]})
print(str(Pb_0))
pred=res_mult_log.get_prediction(Pb_0).summary_frame(alpha=0.05)
print(str(pred))
#pred_B_undre = np.log(10/C)/(k1 - 2*k1_se)
#pred_S_undre = np.log(10/(C*np.exp(k2 - 2*k2_se)))/(k1 - 2*k1_se)
#pred_B_övre = np.log(10/C)/(k1 + 2*k1_se)
#pred_S_övre = np.log(10/(C*np.exp(k2 + 2*k2_se)))/(k1 + 2*k1_se)

#print('Prediktion år blyhalt under 10mg')
#print('Blekinge tidigast: '+str(int(pred_B_undre)+1975))
#print('Södermanland tidigast: '+str(int(pred_S_undre)+1975))
#print('Blekinge senast: '+str(int(pred_B_övre)+1975))
#print('Södermanland senast: '+str(int(pred_S_övre)+1975))

#Prediktion när under 10mg/g mossa 



# %%
