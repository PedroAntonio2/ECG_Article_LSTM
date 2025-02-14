import os
import shutil

def move_images():
    # Caminho base de origem
    base_path = "/home/gpds/Documents/Verify_Article/Results_ECG"

    # Caminho base para o destino reorganizado
    dest_base_path = "/home/gpds/Documents/Verify_Article/Results_ECG_lead"

    # Percorre todos os arquivos do diretório original
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".png"):  # Verifica se é um arquivo de imagem
                # Caminho completo do arquivo atual
                file_path = os.path.join(root, file)
                
                # Divide o caminho original em partes
                parts = file_path.split(os.sep)
                
                # Extrai informações do caminho
                dataset = parts[-6]
                label = parts[-5]
                patient = parts[-4]
                cycle = parts[-3]
                lead = parts[-2]  # Lead correspondente
                
                # Monta o novo caminho de destino
                new_dir = os.path.join(dest_base_path, lead, dataset, label, patient, cycle)
                
                # Cria os diretórios no destino, se necessário
                os.makedirs(new_dir, exist_ok=True)
                
                # Copia o arquivo para o novo local
                shutil.copy(file_path, os.path.join(new_dir, file))

    print("Arquivos reorganizados com sucesso!")

def move_csv():
    # Caminho base de origem
    base_path = "/home/gpds/Documents/Verify_Article/ECG_Data"

    # Caminho base para o destino reorganizado
    dest_base_path = "/home/gpds/Documents/Verify_Article/ECG_Data_lead"

    # Percorre todos os arquivos do diretório original
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".csv"):  # Verifica se é um arquivo de csv
                # Caminho completo do arquivo atual
                file_path = os.path.join(root, file)
                
                # Divide o caminho original em partes
                parts = file_path.split(os.sep)
                
                # Extrai informações do caminho
                dataset = parts[-6]
                label = parts[-5]
                patient = parts[-4]
                cycle = parts[-3]
                lead = parts[-2]  # Lead correspondente
                
                # Monta o novo caminho de destino
                new_dir = os.path.join(dest_base_path, lead, dataset, label, patient, cycle)
                
                # Cria os diretórios no destino, se necessário
                os.makedirs(new_dir, exist_ok=True)
                
                # Copia o arquivo para o novo local
                shutil.copy(file_path, os.path.join(new_dir, file))

    print("Arquivos reorganizados com sucesso!")

if __name__ == "__main__":
    #move_images()
    move_csv()