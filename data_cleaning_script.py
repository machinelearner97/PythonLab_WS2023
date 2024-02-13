
import pandas as pd

def clean_data_and_save(file_path, output_file_path):
    # Load the Excel file
    data = pd.read_excel(file_path)

    # Set the first row as the header
    data.columns = data.iloc[0]
    data = data[1:]

    # Rename the first column
    data = data.rename(columns={'Public ID': 'Sample ID'})

    # Data type conversion - converting all NPX values to numeric
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # Handling missing values
    if data.isnull().sum().sum() > 0:
        print("Warning: There are missing values in the dataset.")
        
    # Save cleaned data to a new Excel file
    data.to_excel(output_file_path, index=False)

    print(f"Cleaned data saved to: {output_file_path}")
    
    return data

# Example usage
file_path = 'K:\\TU_BRAUNSCHWEIG\\PYTHON\\Quaranteam\\TRY1\\NPXvalues.xlsx'
output_file_path = 'K:\\TU_BRAUNSCHWEIG\\PYTHON\\Quaranteam\\TRY1\\Cleaned_NPXvalues.xlsx'
clean_data_and_save(file_path, output_file_path)
