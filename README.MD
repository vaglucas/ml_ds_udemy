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
Census
-------------------
0.7559 - Base line classifier (ZeroR)

0.4767 - Naive Bayes (labelencoder + onehotenconder + escalonamento)
0.7952 - Naive Bayes (labelencoder)
0.7950 - Naive Bayes (labelencoder + onehotencoder)
0.8057 - Naive Bayes (labelencoder + escalonamento)

0.8102 - Árvore de decisão (labelencoder + onehotencoder + escalonamento)
0.8128 - Árvore de decisão (labelencoder)
0.8102 - Árvore de decisão (labelencoder + onehotencoder)
0.8128 - Árvore de decisão (labelencoder + escalonamento)

0.8476 - Random forest n_estimators = 40 (labelencoder + onehotencoder + escalonamento)
0.8481 - Random forest n_estimators = 40 (labelencoder)
0.8489 - Random forest n_estimators = 40 (labelencoder + onehotencoder)
0.8479 - Random forest n_estimators = 40 (labelencoder + escalonamento)
0.8321 - Random forest n_estimators = 40 (labelencoder) + PCA 6 componentes
0.8200 - Random forest n_estimators = 40 (labelencoder) + KernelPCA 6 componentes
0.7334 - Random forest n_estimators = 40 (labelencoder) + LDA 1 componente

0.7880 - CN2 (sem pré-processamento)

0.8223 - kNN 5 (labelencoder + onehotencoder + escalonamento)
0.7746 - kNN 5 (labelencoder)
0.7760 - kNN 5 (labelencoder + onehotencoder)
0.8219 - kNN 5 (labelencoder + escalonamento)

0.8495 - Regressão logística (labelencoder + onehotencoder + escalonamento)
0.7909 - Regressão logística (labelencoder)
0.7954 - Regressão logística (labelencoder + onehotencoder)
0.8184 - Regressão logística (labelencoder + escalonamento)

0.8507 - SVM linear (labelencoder + onehotencoder + escalonamento)
?      - SVM linear (labelencoder)
?      - SVM linear (labelencoder + onehotencoder)
0.8507 - SVM linear (labelencoder + escalonamento)

0.8352 - Rede neural scikit-learn (labelencoder + onehotencoder + escalonamento)
0.2440 - Rede neural scikit-learn (labelencoder)
0.7881 - Rede neural scikit-learn (labelencoder + onehotencoder)
0.8481 - Rede neural scikit-learn (labelencoder + escalonamento)

0.8178 - Rede neural keras (labelencoder + onehotencoder + escalonamento)
0.7559 - Rede neural keras (labelencoder)
0.2440 - Rede neural keras (labelencoder + onehotencoder)
0.8511 - Rede neural keras (labelencoder + escalonamento)

Credit Data
--------------------
0.8720 - Base line classifier
0.9379 - Naive Bayes (inconsistentes + faltantes + escalonamento)
0.9819 - Árvore de decisão (inconsistentes + faltantes + escalonamento)
0.9839 - Random forest n_estimators = 40 (inconsistentes + faltantes + escalonamento)
0.9520 - CN2 (sem pré-processamento)
0.9859 - kNN (inconsistentes + faltantes + escalonamento)
0.9439 - Regressão logística (inconsistentes + faltantes + escalonamento)
0.9879 - SVM kernel rbf cost 2 (inconsistentes + faltantes + escalonamento)
0.9960 - Rede neural scikit-learn (inconsistentes + faltantes + escalonamento)
0.9480 - Rede neural keras (inconsistentes + faltantes + escalonamento)
Naive baiese
