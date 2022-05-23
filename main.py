from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import json

file = open('dataset.json')
data = json.load(file)
dataset = pd.DataFrame.from_dict(data)

# -----------------------------------------------------------------------
# Pré-processamento do alvo e das variáveis de entrada
# -----------------------------------------------------------------------
dataset['DETACH'] = [1 if not np.isnan(int(x)) and int(x) > 365 else 0 for x in dataset['unemployed_time']]  # pessoas que tiveram mais de 1 ano de serviço

dataset['PRE_AGE'] = [16 if np.isnan(int(x)) or x < 16 else x for x in dataset['age']]
dataset['PRE_AGE'] = [1 if x > 65 else (x - 16) / (65 - 16) for x in dataset['PRE_AGE']]
dataset['PRE_GENDER_M'] = [1 if x == 'male' else 0 for x in dataset['gender']]
dataset['PRE_EMPLOYEE'] = [1 if x == 'yes' else 0 for x in dataset['still_work']]  # se ainda trabalha
dataset['PRE_CHANGE'] = [1 if x == 'yes' else 0 for x in dataset['possibility_of_change']]  # se tem possibilidade de mudança
dataset['PRE_REMOTE'] = [1 if x == 'yes' else 0 for x in dataset['is_remote']]  # se tem possibilidate de trabalhar remoto
dataset['PRE_WORK_REGIME_PJ'] = [1 if x == 'PJ' else 0 for x in dataset['work_regime']]  # regime PJ
dataset['PRE_CNH'] = [1 if x == 'yes' else 0 for x in dataset['has_cnh']]  # se tem CNH
dataset['PRE_ENGLISH'] = [1 if x == 'yes' else 0 for x in dataset['speak_english']]  # se fala inglês
dataset['PRE_SPANISH'] = [1 if x == 'yes' else 0 for x in dataset['speak_spanish']]  # se fala espanhol
dataset['PRE_SCHOOL_MERIT'] = [1 if x == 'yes' else 0 for x in dataset['school_merit']]  # se tem mérito escolar
dataset['PRE_TRAVEL'] = [1 if x == 'yes' else 0 for x in dataset['travel_availability']]  # se avalia viagens
dataset['PRE_CRIMINAL'] = [1 if x == 'yes' else 0 for x in dataset['criminal_record']]  # se tem antecedentes criminais
dataset['PRE_RECOMMENDATION'] = [1 if x == 'yes' else 0 for x in dataset['recommendation']]  # se foi indicação

# -----------------------------------------------------------------------
# Separando dados de treinamento e teste
# -----------------------------------------------------------------------


cols_in = [
    'PRE_AGE',
    'PRE_GENDER_M',
    'PRE_EMPLOYEE',
    'PRE_CHANGE',
    'PRE_REMOTE',
    'PRE_WORK_REGIME_PJ',
    'PRE_CNH',
    'PRE_ENGLISH',
    'PRE_SPANISH',
    'PRE_SCHOOL_MERIT',
    'PRE_TRAVEL',
    'PRE_CRIMINAL',
    'PRE_RECOMMENDATION'
]

y = dataset['DETACH']
x = dataset[cols_in]

SEED = 250
np.random.seed(SEED)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.25)

# -----------------------------------------------------------------------
# Gerando modelo
# -----------------------------------------------------------------------


model = KNeighborsClassifier()
model.fit(train_x, train_y)

# -----------------------------------------------------------------------
# Previsão
# -----------------------------------------------------------------------

predicts = model.predict(test_x)

# -----------------------------------------------------------------------
# Calculando taxa de acerto
# -----------------------------------------------------------------------

accuracy = accuracy_score(test_y, predicts)

print('Taxa de acerto %.2f%%' % (accuracy * 100))
print('Taxa de erro %.2f%%' % ((1 - accuracy) * 100))
