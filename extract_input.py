import os
import numpy as np
import pandas as pd

def extract_input_from_lead(path = "ECG_Data_Lead/2"):
    curDir = os.getcwd()
    # Percorre todos os arquivos do diretório original
    dataPath = os.path.join(curDir, path)

    TRAIN = []
    VAL = []
    TEST = []

    def read_csv(dataset, label, patient, cycle, signal):
        signal = list(signal.iloc[:, 0].values)
        signal.extend([dataset, label, patient, cycle])
        if dataset == 'train':
            TRAIN.append(signal)
        elif dataset == 'val':
            VAL.append(signal)
        else:
            TEST.append(signal)

    for dataset in os.listdir(dataPath):
        for label in os.listdir(os.path.join(dataPath, dataset)):
            for patient in os.listdir(os.path.join(dataPath, dataset, label)):
                for cycle in os.listdir(os.path.join(dataPath, dataset, label, patient)):
                    for sinal in os.listdir(os.path.join(dataPath, dataset, label, patient, cycle)):
                        pathToSignal = os.path.join(dataPath, dataset, label, patient, cycle, sinal)
                        signal = pd.read_csv(pathToSignal, header=None)
                        read_csv(dataset, label, patient, cycle, signal)
            
    train = pd.DataFrame(TRAIN)
    val = pd.DataFrame(VAL)
    test = pd.DataFrame(TEST)

    # Rename the columns 700 701 702 703
    train.rename(columns={700: 'dataset', 701: 'label', 702: 'patient', 703: 'cycle'}, inplace=True)
    val.rename(columns={700: 'dataset', 701: 'label', 702: 'patient', 703: 'cycle'}, inplace=True)
    test.rename(columns={700: 'dataset', 701: 'label', 702: 'patient', 703: 'cycle'}, inplace=True)

    # Save the dataframes to csv
    train.to_csv('ECG_Data_Lead_csv/train.csv', index=False)
    val.to_csv('ECG_Data_Lead_csv/val.csv', index=False)
    test.to_csv('ECG_Data_Lead_csv/test.csv', index=False)

def extract_input(path = 'ECG_Data_Lead_11_labels'):
    curDir = os.getcwd()
    # Percorre todos os arquivos do diretório original
    dataPath = os.path.join(curDir, path)

    TRAIN = []
    VAL = []
    TEST = []

    def read_csv(dataset, label, patient, cycle, signal):
        signal = list(signal.iloc[:, 0].values)
        signal.extend([dataset, label, patient, cycle])
        if dataset == 'train':
            TRAIN.append(signal)
        elif dataset == 'val':
            VAL.append(signal)
        else:
            TEST.append(signal)

    for dataset in os.listdir(dataPath):
        for label in os.listdir(os.path.join(dataPath, dataset)):
            for patient in os.listdir(os.path.join(dataPath, dataset, label)):
                for cycle in os.listdir(os.path.join(dataPath, dataset, label, patient)):
                    for sinal in os.listdir(os.path.join(dataPath, dataset, label, patient, cycle)):
                        pathToSignal = os.path.join(dataPath, dataset, label, patient, cycle, sinal)
                        signal = pd.read_csv(pathToSignal, header=None)
                        read_csv(dataset, label, patient, cycle, signal)
            
    train = pd.DataFrame(TRAIN)
    val = pd.DataFrame(VAL)
    test = pd.DataFrame(TEST)

    # Rename the columns 700 701 702 703
    train.rename(columns={700: 'dataset', 701: 'label', 702: 'patient', 703: 'cycle'}, inplace=True)
    val.rename(columns={700: 'dataset', 701: 'label', 702: 'patient', 703: 'cycle'}, inplace=True)
    test.rename(columns={700: 'dataset', 701: 'label', 702: 'patient', 703: 'cycle'}, inplace=True)

    # Save the dataframes to csv
    train.to_csv('ECG_Data_Lead_csv/train.csv', index=False)
    val.to_csv('ECG_Data_Lead_csv/val.csv', index=False)
    test.to_csv('ECG_Data_Lead_csv/test.csv', index=False)
