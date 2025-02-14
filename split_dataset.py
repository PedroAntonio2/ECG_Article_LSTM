import pandas as pd
from sklearn.model_selection import train_test_split
import os
import gc

# Carregar os dados
path = os.path.join(os.getcwd(), 'ECG_Data_Lead_csv/11_labels/ECG_11_labels_no_splits.csv')
df = pd.read_csv(path)
df = df.dropna()
df['patient'] = df['patient'].str.extract('(\d+)').astype(int)

# Definir as classes que devem ficar apenas no treinamento
exclusive_train_classes = [6, 8, 9, 10]

# Separar os dados das classes exclusivas para treino
train_only_df = df[df['label'].isin(exclusive_train_classes)]

# Separar os dados das outras classes que serão divididas entre treino, validação e teste
remaining_df = df[~df['label'].isin(exclusive_train_classes)]

# Garantir que pacientes estejam apenas em um dos conjuntos
patients = remaining_df['patient'].unique()
train_patients, temp_patients = train_test_split(patients, test_size=0.3, random_state=490)  # 70% treino, 30% temp
val_patients, test_patients = train_test_split(temp_patients, test_size=0.5, random_state=490)  # 50% validação, 50% teste

# Criar os datasets garantindo a separação por pacientes
train_df = remaining_df[remaining_df['patient'].isin(train_patients)]
validation_df = remaining_df[remaining_df['patient'].isin(val_patients)]
test_df = remaining_df[remaining_df['patient'].isin(test_patients)]

# Adicionar os dados exclusivos do treino
train_df = pd.concat([train_df, train_only_df]).sample(frac=1, random_state=490)  # Embaralha os dados

# Salvar os arquivos
train_df.to_csv('ECG_Data_Lead_csv/11_labels/ECG_11_labels_correct_train.csv', index=False)
validation_df.to_csv('ECG_Data_Lead_csv/11_labels/ECG_11_labels_correct_val.csv', index=False)
test_df.to_csv('ECG_Data_Lead_csv/11_labels/ECG_11_labels_correct_test.csv', index=False)

# Liberar memória
del train_df, validation_df, test_df, train_only_df, remaining_df, df
gc.collect()
