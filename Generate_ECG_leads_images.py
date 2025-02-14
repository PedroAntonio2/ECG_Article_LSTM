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

            os.makedirs(f'/home/gpds/Documents/Verify_Article/Results_ECG/{dataset}/{currentSignalLabel}/{currentSignalName}/{cycle}/{count}', exist_ok=True)
            Display(ecg_lead_signal, FIGSIZE, PLOT_LIMS).save_lead_image(f'/home/gpds/Documents/Verify_Article/Results_ECG/{dataset}/{currentSignalLabel}/{currentSignalName}/{cycle}/{count}/img_cycle.png')
        print(f"Cycle {cycle}/{ecg.rPeakIndexes.shape[0] - 1} processed")

    except:
        count += 1
        errors_list_position.append(count)
        print(f'Error in signal {currentSignalName} in lead {count}')

def processSignals():
    count = 0
    start = time.time()
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
    processSignals()

    print(f'Time: {time.time() - start} seconds')