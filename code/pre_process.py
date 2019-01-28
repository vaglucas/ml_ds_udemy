import pandas as pd
import  numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
base = pd.read_csv('credit-data.csv')
#print(base.describe())
#print(base.loc[base['age']<0])

#apagar a coluna
#base.drop('age', 1, inplace=True )


#apagar somente os registros com problema
#base.drop(base[base.age< 0].index, inplace=True)

#preencher os valores manualmente

#preencher os valores com a media
#print(base.mean())

#print(base['age'][base.age > 0].mean())

base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

#print(base.loc[pd.isnull(base['age'])])

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

print(previsores)

imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean', )
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])

scaler = StandardScaler()
#padronizacao dos valores, uma menor diferenca entre os valore x = (x - media(x))/desvio padrao(x)
previsores = scaler.fit_transform(previsores)
print(previsores[:,0] )
previsores_treinamento, previsorest_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)


