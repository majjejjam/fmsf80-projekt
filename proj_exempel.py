# -*- coding: utf-8 -*-

#%% importera moduler
import numpy as np
import scipy.stats as stats
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

#%% läs in data
Pb = pd.read_csv('Pb.csv', encoding='utf-8')

#convertera kolumner med Län till en kategorisk variabel
Pb.Lan = Pb.Lan.astype('category')
#Vi kan också addera en kolumn som är år sedan 1975 för att få en lite bättre
#X matris
Pb['Year1975'] = Pb.Year - 1975

#sammanfattning av data
print( Pb )
#En vanlig descibe ger bara sammanfattning för numeriska kolumner
print( Pb.describe() )
#vi behöver be om sammanfattning för alla kolumner
print( Pb.describe(include='all') )
#vi kan också räkna förekomster av de två olika länen
print( Pb.Lan.value_counts() )

#%% Plotta data
sns.scatterplot(Pb, x='Year', y='Pb')
#eller färgkodat per Län
plt.figure()
sns.scatterplot(Pb, x='Year', y='Pb', hue='Lan')
plt.show()

#plt.xscale('log')och/eller plt.yscale('log') kan användas för att
#plotta i log skalor

#%% plocka ut data från ett län
Pb_B = Pb.query("Lan=='Blekinge län'")
Pb_S = Pb.query("Lan=='Södermanlands län'")

#Kontrollera att vi plockat ut rätt bitar
print( Pb_B.describe(include='all') )
print( Pb_S.describe(include='all') )

#%% det finns minst två sätt att göra regression i python
#Funktioner från scikit-learn som använder matris formerna direkt och
#kräver att användaren konstruera X matrisen
# sklearn.linear_model.LinearRegression()

#Ett bättre alternativ är statsmodels regressions funktioner som direkt
#omvandlar formler till lämpliga X och Y matriser.
# statsmodels.formula.api.ols()
help( smf.ols )

#enkel regression med statsmodel
res1 = smf.ols(formula='Pb ~ Year', data=Pb_S)
print( res1.fit().summary() )
#Notera att smf.ols själv lägger till ett intercept så
# Pb ~ Year ger modellen Pb = b0 + b1*Year

#Notera ockås att python klagar på konditions nummert för matrisen. Vilket i
#det här fallet är ganska omotiverat. Men om vi använder år från 1975 istället
#blir den gladare (varför?)

res2 = smf.ols(formula='Pb ~ Year1975', data=Pb_S)
print( res2.fit().summary() )
#Notera att ni får samma lutning men ett annat intercept.

#Den kvar varande noten handlar om att residualerna inte är normalfördelade,
#För exakt normalfördelade residualer bör vi ha
# Prob(Omnibus) > 0.05
# Skew=0
# Kurtosis=1
# Prob(JB) > 0.05
#även här är Python anningen pettig; regressionen fungerar rimligt bra även för
#mindre avvikelser från normalfördelade residualer.

#vill vi trasformera y eller x värden kan transformen tas med i formlen,
#tänk på att de flesta relevanta funktioner finns i numpy. Exempel
# res2 = smf.ols(formula='np.exp(Pb) ~ Year1975', data=Pb_S)

#%% Vi vill nu analysera resultatet av regressionen.
#plocka ut skattade parametrar
beta = res2.fit().params

#plotta data och anpassade linje
sns.scatterplot(Pb_S, x='Year1975', y='Pb')
plt.axline((0,beta.Intercept), slope=beta.Year1975) #point starting in (0,p) with 0 slope
plt.xlabel('Years since 1975')

#%% alternativt skappa en kolumn(!) vektor (behövs för mer komplicerade modeller)
year = np.linspace(min(Pb_S.Year1975), max(Pb_S.Year1975))
#beräkna prediktion för varje värde längs vektorn
Pb_pred = beta[0] + beta[1]*year
#plotta
sns.scatterplot(Pb_S, x='Year1975', y='Pb')
plt.plot(year, Pb_pred)
plt.xlabel('Years since 1975')
plt.show()

#%% eller mot ursprungliga datum
year = np.linspace(min(Pb_S.Year), max(Pb_S.Year))
#beräkna prediktion för varje värde längs vektorn
Pb_pred = beta[0] + beta[1]*(year-1975)
#plotta
sns.scatterplot(Pb_S, x='Year', y='Pb')
plt.plot(year, Pb_pred)
plt.show()

#%% vi kan nu plocka ut lite olika värden
#skattade parameterar
res1.fit().params
#konfidens intervall för parametrarna
res1.fit().conf_int()
#Kovariansmatris för parametrarna
res1.fit().cov_params()
#anpassade värden
yhat = res1.fit().fittedvalues
#residualer
epsilon = res1.fit().resid

#%% Gör projektet
# 1) Gör en residual analys och fundera på vilken modell ni vill använda
#    Bra funktioner att använda här är
#      sns.histplot(x=epsilon, stat='density', kde=True)
#      stats.probplot(epsilon, fit=True, plot=plt)
#      sns.scatterplot(x=Pb_S.Year1975, y=epsilon)
#      sns.residplot(x=yhat, y='Pb', data=Pb_S, lowess=True)
#
# 2) För regressionen på hela materialet behöver ni fundera på formeln i
#    fitlm. I princip
#    Y ~ A + B     ger modellen Y = b0 + b1*A + b2*B
#    Y ~ A*B       ger modellen Y = b0 + b1*A + b2*B + b3*(A.*B)
#    Y ~ A:B       ger modellen Y = b0 + b1*(A.*B)
#    Y ~ A + A:B   ger modellen Y = b0 + b1*A + b2*(A.*B)
#    Pröva de olika formlerna titta på vilka coefficenter som skattas och
#    fundera på vilka modeller det svarar mot.
#
# 3) Det finns en function för prediktion i linjära modeller
#    https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.predict.html
#    https://www.statsmodels.org/dev/dev/generated/statsmodels.base.model.GenericLikelihoodModelResults.get_prediction.html
#    https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.PredictionResults.summary_frame.html
#    https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.PredictionResults.conf_int.html
#    För att använda predikt funktionen behöver vi konstruera en dataframe med
#    värden där vi vill prediktera.
#      Pb_0 = pd.DataFrame({'Year1975' : [?, ?],
#                           'Lan' : [?, ?]})
#    Punkt prediktioner
#      res.fit().predict(Pb_0)
#    Konfidens eller prediktionsintervall
#      res.fit().get_prediction(Pb_0).conf_int(obs=True/False, alpha=?)
#    Allting på en gång
#      res.fit().get_prediction(Pb_0).summary_frame(alpha=0.05)
#    Ni kan behöva titta på alternativen (obs och alpha) för att
#    bestämma signifikansnivå på intervallet och typ av intervall.
#
# 4) Jämför gärna med vad ni får om ni bara använder de skattade
#    beta-värdena från regression för att beräkna ett predikterat y.
#
# 5) För sista uppgiften är följande funktioner bra att tänka på
# res1.fit().cov_params() - Kovariansmatris för beta-skattningen.
# stats.multivariate_normal.rvs - Simulerar från beroende normal variabler
#                                 (MultiVariate Normal) givet vektor med
#                                 väntevärden och Kovariansmatris.
# np.quantile - Hitta kvantiler i en vektor av tal (för att hitta gränser till
#               intervallet)
