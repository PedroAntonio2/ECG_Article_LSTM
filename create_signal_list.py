import pandas as pd
import os
import time

def ptb_diagnostic_create_signal_list_2_labels_(path = 'ptb-diagnostic-ecg-database'):
    signals = {'path': [], 'labels': []}
    count_MI = 0
    count_HC = 0
    database = f"/home/gpds/Documents/Verify_Article/{path}"
    for patient in os.listdir(database):
        patient_path = os.path.join(database, patient)
        for file in os.listdir(patient_path):
            if file.endswith('.hea'):
                header_path = os.path.join(patient_path, file)
                with open(header_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        if line.startswith("# Reason for admission:"):
                            label = line.split(":", 1)[1].strip()
                            if label == 'Myocardial infarction':
                                signal_label = '1'
                                count_MI+=1
                            elif label == 'Healthy control':
                                signal_label = '0'
                                count_HC+=1
                            else:
                                label = 'Other'
                                break
                if label != 'Other':
                    signal_path = header_path.replace('.hea', '.dat')
                    signals['path'].append(signal_path)
                    signals['labels'].append(signal_label)
                    
                    
                break
    print(f"MI: {count_MI}")
    print(f"HC: {count_HC}")
    total = {'MI': count_MI, 'HC': count_HC, 'total': count_MI+count_HC}

def create_signal_list_11_labels(path = 'ptb-diagnostic-ecg-database'):
    signals = {'path': [], 'labels': []}
    count_MI = 0
    count_labels = {}
    count_HC = 0
    database = f"/home/gpds/Documents/Verify_Article/{path}"

    infarction_labels = dict()

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

    for patient in os.listdir(database):
        patient_path = os.path.join(database, patient)
        for file in os.listdir(patient_path):
            if file.endswith('.hea'):
                header_path = os.path.join(patient_path, file)
                with open(header_path, 'r') as file:
                    lines = file.readlines()
                    for idx, line in enumerate(lines):
                        if line.startswith("# Reason for admission:"):
                            label = line.split(":", 1)[1].strip()
                            if label == 'Myocardial infarction':
                                count_MI+=1
                                if idx + 1 < len(lines) and lines[idx + 1].startswith("# Acute infarction"):
                                    infarction_type = lines[idx + 1].split(":", 1)[1].strip()
                                    infarction_labels[infarction_type] = infarction_labels.get(infarction_type, 0) + 1
                                    if infarction_type == 'no':
                                        print(patient_path)
                                    elif infarction_type == 'infero-latera':
                                        infarction_type = 'infero-lateral'
                                    elif infarction_type == 'infero-poster-lateral':
                                        infarction_type = 'infero-postero-lateral'
                                    for key, value in infarction_types.items():
                                        if infarction_type.lower() == key:
                                            signal_label = str(value)
                                            count_labels[key] = count_labels.get(key, 0) + 1
                                            break 

                            elif label == 'Healthy control':
                                signal_label = '0'
                                count_labels['healthy'] = count_labels.get('healthy', 0) + 1
                                count_HC+=1
                            else:
                                label = 'Other'
                                break
                if label != 'Other' and infarction_type != 'no':
                    signal_path = header_path.replace('.hea', '.dat')
                    signals['path'].append(signal_path)
                    signals['labels'].append(signal_label)
                    
                    
                break
    print(f"MI: {count_MI}")
    print(f"HC: {count_HC}")
    total = {'MI': count_MI, 'HC': count_HC, 'total': count_MI+count_HC}
    
    df = pd.DataFrame(signals['path'], columns=['path'])
    df['label'] = signals['labels']
    #df.to_csv('signals_list.csv', index=False)
    split_csv_train_val_test(df, '11_labels')

def split_csv_train_val_test(signal_list, labels = '2_labels'):
    from sklearn.model_selection import train_test_split
    #70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(signal_list, test_size=0.3, random_state=490)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=490)

    if labels == '11_labels':
        train_df.to_csv( index=False)
        val_df.to_csv('signals_list_val_11_labels.csv', index=False)
        test_df.to_csv('signals_list_test_11_labels.csv', index=False)

    elif labels == '2_labels':
        train_df.to_csv('signals_list_train.csv', index=False)
        val_df.to_csv('signals_list_val.csv', index=False)
        test_df.to_csv('signals_list_test.csv', index=False)

    print("Dataset dividido")
        


if(__name__ == '__main__'):
    start = time.time()
    #create_signal_list_2_labels()
    create_signal_list_11_labels()

    print(f"Time: {time.time() - start}")
