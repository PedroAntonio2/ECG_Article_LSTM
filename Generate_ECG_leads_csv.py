from src.signal.Read import Read
from src.signal.Signal import ECG
from src.filter.Filter import Filter
from src.display.Display import Display
from src.processing.SignalProcess import SignalProcess
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

FIGSIZE = (2.24, 2.24)
datasets = ['train', 'val', 'test']

count_MI = 0
count_HC = 0

errors_list_position = []

def processECG(ecg, cycle, dataset, currentSignalLabel, currentSignalName):
    try:
        count = 0
        for ecg_lead in range(ecg.signal.shape[1]):
            count+= 1
            #Normalizing the signal
            max_value = np.max(np.abs(ecg.signal[:, ecg_lead]))
            ecg_lead_signal = ecg.signal[:, ecg_lead]/max_value

            #Taking the 250 samples before the R peak and 450 samples after the R peak
            #ecg_lead_signal = ecg_lead_signal[max(0,rPeakIndexes-250):rPeakIndexes[cycle-1]+450, ecg_lead]
            PLOT_LIMS = [max(0, ecg.rPeakIndexes[cycle-1]-250), max(0, ecg.rPeakIndexes[cycle-1]+450), -1, 1]

            """
            #Removing noise
            wavelet = 'sym5'
            coeffs = pywt.wavedec(ecg_lead, wavelet, level=4)

            threshold=1
            for i in range(1, len(coeffs)):
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
            ecg_denoised = pywt.waverec(coeffs,wavelet)
            """
            # Recortando os dados do ciclo (700 valores)
            start_idx = max(0, ecg.rPeakIndexes[cycle - 1] - 250)
            end_idx = ecg.rPeakIndexes[cycle - 1] + 450
            ecg_lead_signal = ecg.signal[start_idx:end_idx, ecg_lead]

            # Salvar os dados em CSV
            save_ecg_to_csv(ecg_lead_signal, dataset, currentSignalLabel, currentSignalName, count, cycle)
        
            
        print(f"Cycle {cycle}/{ecg.rPeakIndexes.shape[0] - 1} processed")

    except:
        count += 1
        errors_list_position.append(count)
        print(f'Error in signal {currentSignalName} in lead {count}')

def save_ecg_to_csv(ecg_lead_signal, dataset, currentSignalLabel, currentSignalName, lead_count, cycle):
    """
    Função para salvar os dados do ECG em um CSV. 
    Ela irá criar um arquivo para cada ciclo, derivação, e sinal do paciente.
    """
    # Prepara o nome do arquivo e o diretório para salvar
    if dataset == 'undefined':
        output_dir = f'/home/gpds/Documents/Verify_Article/ECG_Data_Lead_11_labels/{currentSignalLabel}/{currentSignalName}/{cycle}/{lead_count}'
    else:
        output_dir = f'/home/gpds/Documents/Verify_Article/ECG_Data_Lead/{dataset}/{currentSignalLabel}/{currentSignalName}/{cycle}/{lead_count}'
    os.makedirs(output_dir, exist_ok=True)

    # Define o nome do arquivo CSV
    csv_filename = f'sinal.csv'
    csv_filepath = os.path.join(output_dir, csv_filename)

    # Converte o ecg_lead_signal para um DataFrame (uma linha por valor)
    df = pd.DataFrame(ecg_lead_signal)
    
    # Salva os dados no arquivo CSV
    df.to_csv(csv_filepath, index=False, header=False)
    #print(f'Saved: {csv_filepath}')

def processSignals11Labels():
    signal_list_train = pd.read_csv('signals_list_train_11_labels.csv')
    signal_list_val = pd.read_csv('signals_list_val_11_labels.csv')
    signal_list_test = pd.read_csv('signals_list_test_11_labels.csv')
    signal_list = pd.concat([signal_list_train, signal_list_val, signal_list_test], ignore_index=True) 

    for numCurrentSignal in range(0, signal_list.shape[0]):
        max_value = 0
        currentSignal = signal_list.iloc[numCurrentSignal, :]
        currentSignalPath = currentSignal['path']
        currentSignalLabel = currentSignal['label']

        ecg = Read.read_ecg_dat(currentSignalPath)
        #ecg.signal = Filter.fir_biosppy(ecg.signal, ecg.samplingFreq)
        
        numberOfCycles = ecg.rPeakIndexes.shape[0] - 1
        for cycle in range(1, numberOfCycles+1):
            processECG(ecg, cycle, 'undefined', currentSignalLabel, f'{currentSignalPath.split("/")[-2]}')

        print(f'Signal {numCurrentSignal+1}/{signal_list.shape[0]}')

def processSignals2Labels():
    for dataset in datasets:
        signal_list = pd.read_csv(f'signals_list_{dataset}.csv')

        for numCurrentSignal in range(0, signal_list.shape[0]):
            max_value = 0
            currentSignal = signal_list.iloc[numCurrentSignal, :]
            currentSignalPath = currentSignal['path']
            currentSignalLabel = currentSignal['label']

            ecg = Read.read_ecg_dat(currentSignalPath)
            ecg.signal = Filter.fir_biosppy(ecg.signal, ecg.samplingFreq)
            
            numberOfCycles = ecg.rPeakIndexes.shape[0] - 1
            for cycle in range(1, numberOfCycles+1):
                processECG(ecg, cycle, dataset, currentSignalLabel, f'{currentSignalPath.split("/")[-2]}')

            print(f'{dataset}: Signal {numCurrentSignal+1}/{signal_list.shape[0]}')

if(__name__ == '__main__'):
    start = time.time()
    #processSignals2Labels()
    processSignals11Labels()

    print(f'Time: {time.time() - start} seconds')