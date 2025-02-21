import os
import numpy as np
import pandas as pd

def extract_input_from_lead(files_path = "ECG_Data_Lead_11_labels/", save_path='ECG_Data_Lead_csv/11_labels/ECG_11_labels_no_splits.csv'):
    curDir = os.getcwd()
    # Percorre todos os arquivos do diretório original
    dataPath = os.path.join(curDir, files_path)

    DF = []

    def read_csv(lead, label, patient, cycle, signal):
        signal = list(signal.iloc[:, 0].values)
        signal.extend([lead, label, patient, cycle])
        
        DF.append(signal)

    for label in os.listdir(dataPath):
        for patient in os.listdir(os.path.join(dataPath, label)):
            for cycle in os.listdir(os.path.join(dataPath, label, patient)):
                for lead in os.listdir(os.path.join(dataPath, label, patient, cycle)):
                    for sinal in os.listdir(os.path.join(dataPath, label, patient, cycle, lead)):
                        pathToSignal = os.path.join(dataPath, label, patient, cycle, lead, sinal)
                        signal = pd.read_csv(pathToSignal, header=None)
                        read_csv(lead, label, patient, cycle, signal)
            
    df = pd.DataFrame(DF)

    # Rename the columns 700 701 702 703
    df.rename(columns={700: 'lead', 701: 'label', 702: 'patient', 703: 'cycle'}, inplace=True)

    # Save the dataframes to csv
    df.to_csv(save_path, index=False)

if __name__ == '__main__':
    extract_input_from_lead()
    
