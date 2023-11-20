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

# print( Pb.describe() )

Pb_S = Pb.loc[Pb['Lan'] == 'Södermanlands län']
Pb_B = Pb.loc[Pb['Lan'] == 'Blekinge län']

T = Pb_S['Year1975'].values
t_reg = sm.add_constant(T)
# Enkel regression, Linjär modell Södermanland
Y = Pb_S['Pb'].values
res_lin = sm.OLS(Y, t_reg).fit()
print(res_lin.summary())

a, b = res_lin.params
epsilon_lin=res_lin.resid

#plt.scatter(T, Y)
sns.scatterplot(x=T, y = Y)
plt.axline((0, a), slope=b)
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.title('Södermanland linjär')
plt.savefig('Grafer/SödermanlandLinjär.png')
plt.show()


# Enkel regression differential model Södermanland
Y_log = np.log(Y)
res_exp = sm.OLS(Y_log, t_reg).fit()
print(res_exp.summary())

C, k = np.exp(res_exp.params[0]), res_exp.params[1]
epsilon_exp=np.exp(res_exp.resid)
t = np.linspace(0, 40, 200)
y = C*np.exp(k*t)

#plt.scatter(T, np.exp(Y_log))
sns.scatterplot(x= T, y =np.exp(Y_log))
plt.plot(t, y)
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.title('Södermanland exponentiell')
plt.savefig('Grafer/SödermanlandExponentiell.png')
plt.show()

#Jämförelse mellan lin och exp
fig,axs=plt.subplots(nrows=1,ncols=2)
axs[0].set_title('Linjär')
axs[1].set_title('Exponentiell')

sns.histplot(x=epsilon_lin,stat='density',kde=True,ax=axs[0])
sns.histplot(x=epsilon_exp,stat='density',kde=True,ax=axs[1])
plt.savefig('Grafer/Histogram.png')
plt.show()

# Multipel Regression Exponentiell
Y_log = np.log(Pb['Pb'].values)
Platser = Pb['Lan'].values
P = [0 if plats == 'Blekinge län' else 1 for plats in Platser]
T = Pb['Year1975'].values
X = list(zip(T, P))

x_reg = sm.add_constant(X)
res_mult_log = sm.OLS(Y_log, x_reg).fit()
print(res_mult_log.summary())
C, k1, k2 = np.exp(res_mult_log.params[0]), res_mult_log.params[1], res_mult_log.params[2]

t = np.linspace(0, 40, 200)
y_exp_S = C*np.exp(t*k1)*np.exp(k2)
y_exp_B = C*np.exp(t*k1)

#plt.scatter(Pb_S['Year1975'].values, Pb_S['Pb'].values, c='blue')
#plt.scatter(Pb_B['Year1975'].values, Pb_B['Pb'].values, c='red')

#Scatterplots med våra exponentiella modeller för respektive län
sns.scatterplot(x=Pb_S['Year1975'].values, y=Pb_S['Pb'].values, c='blue', label = "Södermanland - Exponentiell modell")
sns.scatterplot(x=Pb_B['Year1975'].values, y=Pb_B['Pb'].values, c='red', label = "Blekinge - Exponentiell modell")

#Våran mätdata för respektive län
plt.plot(t, y_exp_S, c='blue', label = "Södermanland - Mätvärden")
plt.plot(t, y_exp_B, c='red', label = "Blekinge - Mätvärden")

#Lägger till rätt titlar på axlar 
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.legend()
plt.title("Exponentiella modeller")

#Sparar och visar grafen med våra exponentiella modeller 
plt.savefig('Grafer/ExpModeller.png')
plt.show()



# Multipel regression exponentiell men där vi tar bort år 20 (chernobyl)
Pb_ny = Pb[Pb['Year1975'] != 20]


# Multipel regression exponentiell men där vi tar bort år 20 och 25 (chernobyl)
Pb_ny = Pb[(Pb['Year1975'] != 20) & (Pb['Year1975'] != 25)]



# Prediktion med intervall

#C, k1, k2 = np.exp(
#    res_mult_log.params[0]), res_mult_log.params[1], res_mult_log.params[2]
#C_se, k1_se, k2_se = np.exp(
#    res_mult_log.bse[0]), res_mult_log.bse[1], res_mult_log.bse[2]
#t = np.linspace(0, 100, 200)

#Våra konfidensintervall för vår multipel regression modell 
intervals=res_mult_log.conf_int()
C_h,k1_h,k2_h=np.exp(intervals[0][1]),intervals[1][1],intervals[2][1]
C_l,k1_l,k2_l=np.exp(intervals[0][0]),intervals[1][0],intervals[2][0]

#Våran 
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

##Grafer med både undre och övre konfidensinvervall för respektive län  
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

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
plt.savefig()
plt.show()

##Våran prediktion av blyvärdet 2025 för respektive län
#Södermanland 
pred_exp_S_undre=exp(C_l,k1_l,k2_l,50)
pred_exp_S = exp(C,k1,k2,50)
pred_exp_S_övre=exp(C_h,k1_h,k2_h,50)

#Blekinge
pred_exp_B_undre=exp(C_l,k1_l,0,50)
pred_exp_B = exp(C,k1,0,50)
pred_exp_B_övre=exp(C_h,k1_h,0,50)

#Presentation av resultat
print('PREDIKTION 2025[Övre, medel,Undre]')
print('Blekinge: '+str(pred_exp_B_övre)+', '+str(pred_exp_B)+', '+str(pred_exp_B_undre))
print('Södermanland: '+str(pred_exp_S_övre)+', '+str(pred_exp_S)+', '+str(pred_exp_S_undre))

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

#Latex-tabeller för våra regressionsresultat
latex_table_lin = res_lin.summary().as_latex()
latex_table_exp = res_exp.summary().as_latex()
latex_table_mult_log = res_mult_log.summary().as_latex()

with open('regression_table_lin.tex', 'w') as f:
    f.write(latex_table_lin)

with open('regression_table_exp.tex', 'w') as f:
    f.write(latex_table_exp)

with open('regression_table_mult_log.tex', 'w') as f:
    f.write(latex_table_mult_log)

