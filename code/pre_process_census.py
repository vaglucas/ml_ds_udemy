import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Normalizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

base = pd.read_csv('census.csv')
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

labelEnconder_previsores= LabelEncoder()

previsores[:,1] = labelEnconder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelEnconder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelEnconder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelEnconder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelEnconder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelEnconder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelEnconder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelEnconder_previsores.fit_transform(previsores[:,13])

onehotenconde = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotenconde.fit_transform(previsores).toarray()

print(previsores[:,15])

label_classe = LabelEncoder()
classe = label_classe.fit_transform(classe)
classe

scale = StandardScaler()
previsores = scale.fit_transform(previsores)

previsores_treinamento, previsorest_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)
