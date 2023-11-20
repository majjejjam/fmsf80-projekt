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
# print(Pb_S)
# print(Pb_B)


# Enkel regression, Linjär modell

Pb_Lan = Pb.loc[Pb['Lan'] == Län]
T = Pb_Lan['Year1975'].values
Y = Pb_Lan['Pb'].values

t_reg = sm.add_constant(T)
res_lin = sm.OLS(Y, t_reg).fit()
print(res_lin.summary())

a, b = res_lin.params
plt.scatter(T, Y)
plt.axline((0, a), slope=b)
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.title(Län)
plt.show()


# Enkel regression differential model
Pb_Lan = Pb.loc[Pb['Lan'] == Län]

T = Pb_Lan['Year1975'].values
t_reg = sm.add_constant(T)
Y = np.log(Pb_Lan['Pb'].values)

res = sm.OLS(Y, t_reg).fit()
print(res.summary())

C, k = np.exp(res.params[0]), res.params[1]
t = np.linspace(0, 40, 200)
y = C*np.exp(k*t)

plt.scatter(T, np.exp(Y))
plt.plot(t, y)
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.title(Län)
plt.show()





# Multipel Regression

Pb_S = Pb.loc[Pb['Lan'] == 'Södermanlands län']
Pb_B = Pb.loc[Pb['Lan'] == 'Blekinge län']

Y_log = np.log(Pb['Pb'].values)
Platser = Pb['Lan'].values
P = [0 if plats == 'Blekinge län' else 1 for plats in Platser]
T = Pb['Year1975'].values
X = list(zip(T, P))

x_reg = sm.add_constant(X)
res_mult_log = sm.OLS(Y_log, x_reg).fit()
print(res_mult_log.summary())
C, k1, k2 = np.exp(
    res_mult_log.params[0]), res_mult_log.params[1], res_mult_log.params[2]

t = np.linspace(0, 40, 200)
y_exp_S = C*np.exp(t*k1)*np.exp(k2)
y_exp_B = C*np.exp(t*k1)

plt.scatter(Pb_S['Year1975'].values, Pb_S['Pb'].values, c='blue')
plt.scatter(Pb_B['Year1975'].values, Pb_B['Pb'].values, c='red')
plt.plot(t, y_exp_S, c='blue')
plt.plot(t, y_exp_B, c='red')
plt.xlabel("Tid (år)")
plt.ylabel("Bly (mg/kg mossa)")
plt.show()



# Multipel regression exponentiell men där vi tar bort år 20 (chernobyl)
Pb_ny = Pb[Pb['Year1975'] != 20]


# Multipel regression exponentiell men där vi tar bort år 20 och 25 (chernobyl)
Pb_ny = Pb[(Pb['Year1975'] != 20) & (Pb['Year1975'] != 25)]



# Prediktion med intervall

Pb_S = Pb.loc[Pb['Lan'] == 'Södermanlands län']
Pb_B = Pb.loc[Pb['Lan'] == 'Blekinge län']

Y_log = np.log(Pb['Pb'].values)
Platser = Pb['Lan'].values
P = [0 if plats == 'Blekinge län' else 1 for plats in Platser]
T = Pb['Year1975'].values
X = list(zip(T, P))

x_reg = sm.add_constant(X)
res_mult_log = sm.OLS(Y_log, x_reg).fit()
print(res_mult_log.summary())
C, k1, k2 = np.exp(
    res_mult_log.params[0]), res_mult_log.params[1], res_mult_log.params[2]
C_se, k1_se, k2_se = np.exp(
    res_mult_log.bse[0]), res_mult_log.bse[1], res_mult_log.bse[2]
t = np.linspace(0, 100, 200)
y_exp_S = C*np.exp(t*k1)*np.exp(k2)
y_exp_B = C*np.exp(t*k1)

df = pd.DataFrame(index=t)
df['Södermanland'] = y_exp_S
df['Blekinge'] = y_exp_B

df['Beta S'] = k1
df['Beta B'] = k2
df['Beta S - stdev'] = k1_se
df['Beta B - stdev'] = k2_se
df['Beta S - undre'] = k1 - 2*k1_se
df['Beta B - undre'] = k2 - 2*k2_se
df['Beta S - övre'] = k1 + 2*k1_se
df['Beta B - övre'] = k2 + 2*k2_se

y_exp_S_undre = C*np.exp(t*df['Beta S - undre']) * \
    np.exp(df['Beta B - undre'])
y_exp_B_undre = C*np.exp(t*df['Beta S - undre'])
y_exp_S_övre = C*np.exp(t*df['Beta S - övre'])*np.exp(df['Beta B - övre'])
y_exp_B_övre = C*np.exp(t*df['Beta S - övre'])

df['Södermanland - Undre'] = y_exp_S_undre
df['Blekinge - Undre'] = y_exp_B_undre
df['Södermanland - Övre'] = y_exp_S_övre
df['Blekinge - Övre'] = y_exp_B_övre

sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))

sns.lineplot(x=t, y=y_exp_S, color='blue',
                label='Södermanland', ax=axes[0])
sns.lineplot(x=t, y=y_exp_S_undre, color='lightblue',
                label='Södermanland undre', ax=axes[0])
sns.lineplot(x=t, y=y_exp_S_övre, color='darkblue',
                label='Södermanland övre', ax=axes[0])
axes[0].set_xlabel("Tid (år)")
axes[0].set_ylabel("Bly (mg/kg mossa)")
axes[0].legend()
axes[0].set_title("Södermanland")

sns.lineplot(x=t, y=y_exp_B, color='red', label='Blekinge', ax=axes[1])
sns.lineplot(x=t, y=y_exp_B_undre, color='lightcoral',
                label='Blekinge undre', ax=axes[1])
sns.lineplot(x=t, y=y_exp_B_övre, color='darkred',
                label='Blekinge övre', ax=axes[1])
axes[1].set_xlabel("Tid (år)")
axes[1].set_ylabel("Bly (mg/kg mossa)")
axes[1].legend()
axes[1].set_title("Blekinge")

plt.tight_layout()
plt.show()

pred_B_undre = np.log(10/C)/(k1 - 2*k1_se)
pred_S_undre = np.log(10/(C*np.exp(k2 - 2*k2_se)))/(k1 - 2*k1_se)
pred_B_övre = np.log(10/C)/(k1 + 2*k1_se)
pred_S_övre = np.log(10/(C*np.exp(k2 + 2*k2_se)))/(k1 + 2*k1_se)

print('Prediktion år blyhalt under 10mg')
print('Blekinge tidigast: '+str(int(pred_B_undre)+1975))
print('Södermanland tidigast: '+str(int(pred_S_undre)+1975))
print('Blekinge senast: '+str(int(pred_B_övre)+1975))
print('Södermanland senast: '+str(int(pred_S_övre)+1975))


