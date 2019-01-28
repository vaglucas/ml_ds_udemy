Naive Bayes

Arvore de decisao

prox aula 44 randon florest



===============*****====================
-Metodos Preditivos:
  classificacao (clientes, risco, previsao doenca)
  regressao (valor do dolar, ganhos com publicidade)
----------
-Metodos Descritivos:
    associacao (compra associada, mercado)
    agrupamento (agrupar conteudos, clientes, produtos)
  -Deteccao de desvios:
    bloqueio de cartao (fraude)
    uso de energia, agua, telefone acima da media fora do padrao
    DESVIOS DE padrao
  -Padroes sequenciais:
    compra de produtos sequenciais (livros, produtos)
    prevensao de doenca (sintomas um padrao, pode gerar doenca)
    navegacao em sites
  -Sumarizacao:
    dados comuns na empresa, uso de serviços

----------
-Aprendizagem supervisionada:
    extracao de caracteristicas, alg, de Aprendizagem (classificacao). modelo aprendido
    extracao de caracteristicas (classificar)

-Aprendizagem nao supervisionada:
    Analize automatica

-Aprendizagem por reforco:
    aprende com experiencia (causa e efeito)

----------
Classificacao:
  Saber o que vc quer classificar
  Classificacao: base de dados historica
  ===============*****====================

Pre preprocessing
 ajuste dos dados:
 -correcao de vaores em zero
 base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

 -label enconder (string em valores de classificacao)
 previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])

 -divisao da base em previsores e classes de classificacao
 previsores = base.iloc[:,1:4].values
 classe = base.iloc[:,4].values
-OneHotEncoder inclui valore, transforma colunas com dados em diversas colunas com 1 (presente), o (falso)
-Escalonamento dos valores trazer para mais perdo da scala valores que estao fora da curva de valores
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

-train_test_split separa as classes e os previsores para teste e treinamento
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

---CLASSIFICAO---


Naive baiese
