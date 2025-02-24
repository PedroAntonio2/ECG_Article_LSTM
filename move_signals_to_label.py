import os
import pandas as pd
import shutil

path = "Results_ECG_lead"
patient_list_path = "signals_list.csv"
patient_list_csv = pd.read_csv(patient_list_path)
patient_list_filepath = patient_list_csv['path'].values
patient_list_label = patient_list_csv['label'].values
patient_list = []
patient_label = []

for patient in patient_list_filepath:
    patient_list.append(patient.split('/')[-2])

for label in patient_list_label:
    patient_label.append(label)

patients = dict()
for name, label in zip(patient_list, patient_label):
    patients[name] = label

patients_detected = set()

for lead in os.listdir(path):
    lead_path = os.path.join(path, lead)
    for split in os.listdir(lead_path):
        split_path = os.path.join(lead_path, split)
        for label in os.listdir(split_path):
            label_path = os.path.join(split_path, label)
            for patient in os.listdir(label_path):
                patient_path = os.path.join(label_path, patient)
                if patient not in patient_list:
                    patients_detected.add(patient)
                    #print(f"Patient {patient} not in the list")
                    #print(f"Path: {label_path}/{patient}")
                else:
                    destination_label_path = f'/home/gpds/Documents/Verify_Article/Results_ECG_11_labels/{lead}/{split}/{patients[patient]}'
                    destination_path = destination_label_path + f'/{patient}'
                    
                    #shutil.copytree(patient_path, destination_path)
                for cycle in os.listdir(patient_path):
                    cycle_path = os.path.join(patient_path, cycle)
                    for file in os.listdir(cycle_path):
                        file_path = os.path.join(cycle_path, file)
                        destination_file_path = destination_path + f'/{cycle}_{file}'
                        if not os.path.exists(destination_path):
                            os.makedirs(destination_path, exist_ok=True)
                            pass
                        shutil.copy(file_path, destination_file_path)
                        

            