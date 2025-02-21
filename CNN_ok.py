import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, LSTM, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import pandas as pd
import time
import numpy as np
from pathlib import Path

start = time.time()
results_leads = []

labels = 11
PATH = f'/home/gpds/Documents/Verify_Article/Results_ECG_lead_metrics/{labels}'
MODEL_NAME = 'CNN_LSTM'

TIME_MODEL_CREATION = datetime.now().strftime("%d_%m_%Y_-%H-%M-%S")

PATH_TO_SAVE_RESULTS = PATH+'/'+'results_'+MODEL_NAME+'/'+TIME_MODEL_CREATION
Path(PATH_TO_SAVE_RESULTS).mkdir(parents=True, exist_ok=True)

# infarction_types = {
#     'anterior': 1,
#     'antero-lateral': 2,
#     'antero-septal': 3,
#     'inferior': 4,
#     'infero-lateral': 5,
#     'infero-posterior': 6,
#     'infero-postero-lateral': 7,
#     'lateral': 8,
#     'posterior': 9,
#     'postero-lateral': 10,
#     'healthy': 0  # Healthy control
# }

infarction_types = {
    'anterior': 1,
    'antero-lateral': 2,
    'antero-septal': 3,
    'inferior': 4,
    'infero-lateral': 5,
    'infero-posterior': 6,
    'infero-postero-lateral': 7,
    'lateral': 8,
    'posterior': 9,
    'postero-lateral': 10,
    'healthy': 0  # Healthy control
}

leads = {
    'I': 1,
    'II': 2,
    'III': 3,
    'aVR': 4,
    'aVL': 5,
    'aVF': 6,
    'V1': 7,
    'V2': 8,
    'V3': 9,
    'V4': 10,
    'V5': 11,
    'V6': 12,
    'V7': 13,
    'V8': 14,
    'V9': 15

}
print("Start")
df_list = []
splits = ['train', 'val', 'test']
for split in splits:
    print(f"Split: {split}")
    file_path = f'/home/gpds/Documents/Verify_Article/ECG_Data_Lead_csv/{labels}_labels/correct_{split}.csv'
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df_list.append(df)

for lead, lead_number in leads.items():
    metrics_leads = {}

    print(f"Derivação: {lead}")

    # Função para carregar e pré-processar os dados
    def load_data(split, lead):
        if split == 'train':
            df_split = df_list[0]
        elif split == 'val':
            df_split = df_list[1]
        elif split == 'test':
            df_split = df_list[2]

        if lead != 0:
            df_lead = df_split[df_split['lead'] == lead]
        
        # Extrair as features (valores de 0 a 699)
        X = df_lead.iloc[:, 0:700].values  # Extrai as primeiras 700 colunas
        
        # Extrair os rótulos (coluna 701)
        y = df_lead.iloc[:, 701].values 
        
        # Reshape os dados para (n_samples, 700, 1)
        #X = np.expand_dims(X, axis=-1)  # Agora X tem a forma (n_samples, 700, 1)
        
        return X.astype(np.float32), y.astype(int)

    # Carregar dados de treino, validação e teste
    X_train, y_train = load_data('train', lead_number)
    X_val, y_val = load_data('val', lead_number)
    X_test, y_test = load_data('test', lead_number)

    # Calcular os pesos das classes
    class_weights = compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(y_train),
        y = y_train
    )
    class_weights = dict(enumerate(class_weights))

    with tf.device('/CPU:0'):
        # Construção do modelo CNN-LSTM
        model = Sequential([
            # Primeira camada convolucional
            Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(700, 1)),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            # Segunda camada convolucional
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            
            # Camada LSTM
            #LSTM(units=64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', implementation=2, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.l2(0.01)),

            # Camada LSTM
            LSTM(units=64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'),


            #LSTM(units=64, return_sequences=False, activation='tanh', seed = 490),
            
            # Flatten para converter em vetor
            Flatten(),
            
            # Primeira camada Dense totalmente conectada
            BatchNormalization(),
            Dense(units=64, activation='relu'),
            Dropout(0.2),
            
            # Camada Dense de saída
            Dense(units=11, activation='softmax')
        ])

    # Compilação do modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    """
    for i in range(X_train.shape[0]):
        X_train[i,:] = X_train[i,:] / np.max(X_train[i,:])
    for i in range(X_val.shape[0]):
        X_val[i,:] = X_val[i,:] / np.max(X_val[i,:])
    for i in range(X_test.shape[0]):
        X_test[i,:] = X_test[i,:] / np.max(X_test[i,:])
    """

    # Calcular a média e o desvio padrão no conjunto de treino
    mean = np.mean(X_train, axis=0)  # Média por cada coluna (canal ou tempo)
    std = np.std(X_train, axis=0)    # Desvio padrão por cada coluna

    # Evitar divisão por zero (caso algum std seja zero)
    std[std == 0] = 1

    # Aplicar normalização (Z-score) a todos os conjuntos
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)
    y_val = np.expand_dims(y_val, axis=-1)

    model.summary()
    #y_train = np.expand_dims(y_train, axis=-1)
    # Treinamento do modelo
    history = model.fit(
        X_train,
        y_train, 
        epochs = 10, 
        batch_size = 64, 
        validation_data = (X_val, y_val),
        #class_weight=class_weights,
        shuffle=True
    )

    # Avaliar o modelo nos dados de teste
    train_loss, train_acc = model.evaluate(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)

    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")
    print(f"Train Loss: {train_loss}")
    print(f"Train Accuracy: {train_acc}")

    model.save(PATH_TO_SAVE_RESULTS+f'/model_{lead}_ville.h5')

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

        # Criar figura do gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Loss Treinamento', color='blue')
    plt.plot(history.history['val_loss'], label='Loss Validação', color='red')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.title('Loss de Treinamento vs Loss de Validação')
    plt.legend()
    plt.grid(False)

    # Caminho para salvar
    loss_plot_path = PATH_TO_SAVE_RESULTS + f'/loss_plot_{lead}.png'
    plt.savefig(loss_plot_path, dpi=300)  # Salvar com alta resolução

    cm = confusion_matrix(y_test, y_pred)
    #Save cm to csv
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(PATH_TO_SAVE_RESULTS+f'/confusion_matrix_{lead}.csv', index=False)

    class_names = [name for name, _ in sorted(infarction_types.items(), key=lambda x: x[1])]

    #fig, ax = plt.subplots(figsize=(12, 10))
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    #disp.plot(cmap=plt.cm.Blues, ax=ax)

    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PATH_TO_SAVE_RESULTS+f'/confusion_matrix_{lead}.png')

    # Mudando para um problema binário para calcular recall e precision
    y_test_binary = np.where(y_test == 0, 0, 1)
    y_pred_binary = np.where(y_pred == 0, 0, 1)

    accuracy = test_acc
    precision = precision_score(y_test_binary, y_pred_binary)
    recall = recall_score(y_test_binary, y_pred_binary)

    # Imprimir as métricas
    print(f"Binary Accuracy: {accuracy}")
    print(f"Binary Precision: {precision}")
    print(f"Binary Recall: {recall}")

    # Salvar as métricas em um arquivo CSV
    results = {
        "Metric": ["Train Accuracy", "Validation Accuracy", "Test Accuracy", "Precision", "Recall"],
        "Value": [history.history['accuracy'][-1], history.history['val_accuracy'][-1], test_acc, precision, recall]
    }

    results_df = pd.DataFrame(results)
    results_df.to_csv(PATH_TO_SAVE_RESULTS+f'/metrics_{lead}.csv', index=False)

    # Salvar o modelo
    model.save(PATH_TO_SAVE_RESULTS+f'/model_{lead}.h5')

    print(f"Tempo total de execução da derivação {lead}: {time.time() - start:.2f} segundos")

    row = {
        "Lead": lead,
        "Train Accuracy": history.history['accuracy'][-1],
        "Validation Accuracy": history.history['val_accuracy'][-1],
        "Test Accuracy": test_acc,
        "Precision": precision,
        "Recall": recall
    }

    results_leads.append(row)

# Salvar as métricas de todas as derivações em um arquivo CSV

results_df = pd.DataFrame(results_leads)
results_df.to_csv(PATH_TO_SAVE_RESULTS+'/metrics_all_leads.csv', index=False)

print(f"Tempo total de execução: {time.time() - start:.2f} segundos")