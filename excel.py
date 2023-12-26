import pandas as pd
import pickle
file_path = 'dataset_2412_17.pkl'
with open(file_path, "rb") as file:
    data = pickle.load(file)
    # Convierte la lista de diccionarios a un DataFrame de Pandas
df = pd.DataFrame(data)
# Ruta donde deseas guardar el archivo Excel
excel_file_path = "dataset_2412_17.xlsx"

# Guarda el DataFrame en un archivo Excel
df.to_excel(excel_file_path, index=False)
