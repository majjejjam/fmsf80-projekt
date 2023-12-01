#%%Import
import numpy as np
import scipy.stats as stats
import statistics
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Globalt mpl-utseende
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.figsize'] = (4.0, 4.0)
mpl.rcParams['figure.autolayout'] = True
mpl.rcParams['font.size'] = 11
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

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

# Plotta datan
sns.scatterplot(Pb, x='Year1975', y='Pb', hue='Lan', legend=False)
# Justera axlarna
plt.yticks([Pb['Pb'].min()] + list(range(25, 176, 25)))
ax = plt.gca()
ax.spines['left'].set_bounds(Pb['Pb'].min(), 175)
ax.spines['bottom'].set_bounds(Pb['Year1975'].min(), Pb['Year1975'].max())
# Lite andrum
ax.set_xlim([-4, ax.get_xlim()[1]])
ax.set_ylim([-4, ax.get_ylim()[1]])
# Etiketter
plt.title("Blyhalt över tid")
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.savefig('Grafer/DataFörBådaLänen.png')
plt.savefig('Grafer/DataFörBådaLänen.jpg')
plt.savefig('Grafer/DataFörBådaLänen.pgf')
plt.show()

#%% Enkel regression, Linjär modell Södermanland

reg_lin=smf.ols(formula='Pb~Year1975',data=Pb_S).fit()
print(reg_lin.summary())

#Våra parametrar för den linjära modellen
alpha, beta = reg_lin.params

#Vår linjära modells residualer
epsilon_lin=reg_lin.resid

#Vi visar våra resulat från den linjära modellen
#plt.scatter(T, Y)
sns.scatterplot(x=T, y = Y)

#Vår linjära ekvation med våra estimerade parametrar
plt.axline((0, alpha), slope=beta)
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.title('Södermanland linjär')
plt.savefig('Grafer/SödermanlandLinjär.png')
plt.savefig('Grafer/SödermanlandLinjär.pgf')
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
for idx, n in enumerate(T):
    epsilon_exp.append(Y[idx] - C*np.exp(k*n))

#Vi plottar jämförelsen mellan våra mätvärden och estimerade värden från den exponentiella modellen
#plt.scatter(T, np.exp(Y_log))
sns.scatterplot(x= T, y =Y)
plt.plot(t, y)
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.title('Södermanland exponentiell')
plt.savefig('Grafer/SödermanlandExponentiell.png')
plt.savefig('Grafer/SödermanlandExponentiell.pgf')
plt.show()

#%%Jämförelse mellan linjär och exponentiell modell
fig,axs=plt.subplots(1, 2, constrained_layout=True)
axs[0].set_title('Linjär')
axs[1].set_title('Exponentiell')

#Histogram över residualer från våra respektive modeller
sns.histplot(x=epsilon_lin,stat='density',kde=True,ax=axs[0])
sns.histplot(x=epsilon_exp,stat='density',kde=True,ax=axs[1])
plt.savefig('Grafer/Histogram.png')
#plt.savefig('Grafer/Histogram.pgf')
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
plt.savefig('Grafer/ExpModeller.pgf')
plt.show()

#%% Prediktion 2025 exp
Pb_0=pd.DataFrame({'Year1975' : [50, 50],'Lan_I' : [0, 1]})
Pred=res_mult_log.get_prediction(Pb_0).summary_frame(alpha=0.05)
print(str(Pred))
Pred_S_2025=[np.exp(Pred['obs_ci_lower'][1]),np.exp(Pred['mean'][1]),np.exp(Pred['obs_ci_upper'][1])]
Pred_B_2025=[np.exp(Pred['obs_ci_lower'][0]),np.exp(Pred['mean'][0]),np.exp(Pred['obs_ci_upper'][0])]

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

#Värdena på våra exponentiella funktioner för respektive län
y_exp_S = np.exp(beta0+beta1)*np.exp(t*(beta2))
y_exp_B = np.exp(beta0)*np.exp(t*beta2)

#Södermanland undre och övre intervall
y_exp_S_u=np.exp(beta_u[0]+beta_u[1])*np.exp(t*(beta_u[2]))
y_exp_S_ö=np.exp(beta_ö[0]+beta_ö[1])*np.exp(t*(beta_ö[2]))
#Blekinge undre och övre intervall
y_exp_B_u=np.exp(beta_u[0])*np.exp(t*beta_u[2])
y_exp_B_ö=np.exp(beta_ö[0])*np.exp(t*beta_ö[2])

##Grafer med både undre och övre konfidensinvervall för respektive län
def plot_exponential(t, u_gräns = 0, ö_gräns = 200):
    t = t[u_gräns:ö_gräns]
    fig,axs=plt.subplots(2, 1, constrained_layout=True)

    #Våra tre respektive modeller för Södermanland
    sns.lineplot(x=t, y=y_exp_S[u_gräns:ö_gräns], color='blue', ax=axs[0])
    #Konfidensintervall
    axs[0].fill_between(t, y_exp_S_u[u_gräns:ö_gräns], y_exp_S_ö[u_gräns:ö_gräns], color='lavender')

    #Namnger våra axlar korrekt
    axs[0].set_xlabel("Tid (år)")
    axs[0].set_ylabel("Bly (mg/kg mossa)")
    axs[0].set_title("Södermanland    ", loc='right', pad=-14)

    #Våra tre respektive modeller för Blekinge
    sns.lineplot(x=t, y=y_exp_B[u_gräns:ö_gräns], color='red', ax=axs[1])
    #Konfidensintervall
    axs[1].fill_between(t, y_exp_B_u[u_gräns:ö_gräns], y_exp_B_ö[u_gräns:ö_gräns], color='mistyrose')

    #Namnger våra axlar korrekt
    axs[1].set_xlabel("Tid (år)")
    axs[1].set_ylabel("Bly (mg/kg mossa)")
    axs[1].set_title("Blekinge    ", loc='right', pad=-24)

    #Justerar och sparar grafen
    horisont = str(u_gräns)+"-"+str(ö_gräns)

    axs[0].set_xlim(t.min(), t.max())
    axs[1].set_xlim(t.min(), t.max())

    if u_gräns >= 50:
        axs[0].fill_between(t, 10, y_exp_S_u[u_gräns:ö_gräns], color='b', alpha=0.3)
        axs[1].fill_between(t[t>x0_B_u-1975], 10, y_exp_B_u[int(np.round(x0_B_u-1975)):ö_gräns], color='r', alpha=0.3)
        axs[0].scatter(x=x0_S-1975, y=10, color='k', marker='|', linewidth=1, zorder=2)
        axs[0].annotate("{:.0f} år".format(np.round(x0_S-1975)),
                     xy=(x0_S-1975, 10),
                     xytext=(+10, +30), textcoords='offset points', fontsize=12,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

        axs[1].scatter(x=x0_B-1975, y=10, color='k', marker='|', linewidth=1, zorder=2)
        axs[1].annotate("{:.0f} år".format(np.round(x0_B-1975)),
                     xy=(x0_B-1975, 10),
                     xytext=(+10, +30), textcoords='offset points', fontsize=12,
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.savefig('Grafer/ExpModellerIntervall'+horisont+'.png')
    plt.savefig('Grafer/ExpModellerIntervall'+horisont+'.pgf')
    plt.show()

plot_exponential(t)
plot_exponential(t,0,41)
plot_exponential(t,50,121)

#%%En tabell med våra resultat för blyhalter 2025
columns = ['Skattat Värde','Kondidensintervall(95%)']
pred_data = [[np.round(Pred_S_2025[1],2),[np.round(Pred_S_2025[0],2),np.round(Pred_S_2025[2],2)]],
        [np.round(Pred_B_2025[1],2),[np.round(Pred_B_2025[0],2),np.round(Pred_B_2025[2],2)]]]
pred_res_2025 = pd.DataFrame(data=pred_data, index=['Y_S','Y_B']
                  ,columns=columns)
latex_prediktioner_2025 = pred_res_2025.to_latex()

#%%En tabell med våra resultat för åren som blyhalten understiger 10 mg/kg
columns = ['Skattat Värde','Kondidensintervall(95%)']
pred_data_10 = [[np.round(pred_S_10mg[1],2),[np.round(pred_S_10mg[0],2),np.round(pred_S_10mg[2],2)]],
        [np.round(pred_B_10mg[1],2),[np.round(pred_B_10mg[0],2),np.round(pred_B_10mg[2],2)]]]
pred_res_10 = pd.DataFrame(data=pred_data_10, index=['Y_S','Y_B']
                  ,columns=columns)
latex_prediktioner_10 = pred_res_10.to_latex()


##Våra latextabeller 
#Våra regressionsresultat för respektive modell med Södermanland
latex_table_lin = reg_lin.summary().as_latex()
latex_table_exp = reg_exp.summary().as_latex()

##Våra jämförelsevärden
# Vår linjära modell
omnibus_prob_lin = reg_lin.summary().tables[2].data[1][1]
skew_lin = reg_lin.summary().tables[2].data[2][1]
kurt_lin = reg_lin.summary().tables[2].data[3][1]
lin_modell_varden = [omnibus_prob_lin, skew_lin, kurt_lin]

# Vår exponentiella modell
omnibus_prob_exp = reg_exp.summary().tables[2].data[1][1]
skew_exp = reg_exp.summary().tables[2].data[2][1]
kurt_exp = reg_exp.summary().tables[2].data[3][1]
exp_modell_varden = [omnibus_prob_exp, skew_exp, kurt_exp]

#Samlar till en dataframe
modell_varden = [lin_modell_varden,exp_modell_varden]
columns = ['P(Omnibus)','Snedhet','Kurtos']
modell_varden_jamf = pd.DataFrame(data=modell_varden, index=['Linjär','Exponentiell']
                  ,columns=columns)
latex_modell_varden = modell_varden_jamf.to_latex()

#Vårt regressionsreesultat för vår multipla regression 
latex_table_mult_log = res_mult_log.summary().as_latex()
betas = np.round(res_mult_log.params.values,3)
stds = np.round(res_mult_log.bse.values,3)
t_stats = np.round(res_mult_log.tvalues.values,3)

#Samlar allt o en dataframe för Latex
data = pd.DataFrame([betas, stds, t_stats])
params = ['Intercept', 'Lan_I', 'Year1975', 'Lan_I:Year1975'] 
cols = ['Estimat', 'StandardAvvikelse', 'T-värde']
data_multlog = pd.DataFrame(data = data.T)
data_multlog.index = params
data_multlog.columns= cols
latex_multlog = data_multlog.to_latex()

#Våra konfidensintervall för våra parametrar i vår multipel regression
cols = ['Undre Intervall', 'Väntevärde', 'Övre Intervall']
df = pd.DataFrame(index= params, columns = cols)
df.iloc[:,0] = beta_u
df.iloc[:,1] = beta
df.iloc[:,2] = beta_ö
latex_konf = df.to_latex()

#Sparar allt till latex
with open('LatexTabeller/regression_table_lin.tex', 'w') as f:
    f.write(latex_table_lin)

with open('LatexTabeller/regression_table_exp.tex', 'w') as f:
    f.write(latex_table_exp)

with open('LatexTabeller/regression_table_mult_log.tex', 'w') as f:
    f.write(latex_table_mult_log)

with open('LatexTabeller/prediktioner_2025.tex', 'w') as f:
    f.write(latex_prediktioner_2025)

with open('LatexTabeller/prediktioner_10.tex', 'w') as f:
    f.write(latex_prediktioner_10)

with open('LatexTabeller/modell_jamforelse.tex', 'w') as f:
    f.write(latex_modell_varden)

with open('LatexTabeller/multlog.tex', 'w') as f:
    f.write(latex_multlog)

with open('LatexTabeller/multlogkond.tex', 'w') as f:
    f.write(latex_konf)

