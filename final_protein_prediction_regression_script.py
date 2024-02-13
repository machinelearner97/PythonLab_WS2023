
import pandas as pd
from sklearn.linear_model import LinearRegression

def clean_and_predict(file_path):
    # Load the data
    data = pd.read_excel(file_path)

    # Extract day information and actual sample ID
    data['Day'] = data['Sample ID'].apply(lambda x: x.split('_')[1])
    data['Sample ID'] = data['Sample ID'].apply(lambda x: x.split('_')[0])

    # Filter out irrelevant days and reshape the data
    filtered_data = data[data['Day'].isin(['D0', 'D3', 'D7'])]
    melted_data = pd.melt(filtered_data, id_vars=['Sample ID', 'Day'], var_name='Protein ID', value_name='NPX Value')
    wide_data = melted_data.pivot_table(index=['Sample ID', 'Protein ID'], columns='Day', values='NPX Value').reset_index()
    wide_data = wide_data.dropna()

    # Preparing a dictionary to store the models and predictions for each protein
    protein_predictions = {}

    # Looping over each protein to create a model and make predictions
    for protein in wide_data['Protein ID'].unique():
        protein_data = wide_data[wide_data['Protein ID'] == protein]
        X = protein_data[['D0', 'D3']]
        y = protein_data['D7']
        
        # Model training
        model = LinearRegression()
        model.fit(X, y)

        # Making predictions for D7
        predictions = model.predict(X)

        # Storing the predictions
        protein_predictions[protein] = predictions

    return protein_predictions

# Example usage
file_path = 'C:/Users/Nandini/Desktop/TRY1/TRY1/Cleaned_NPXvalues.xlsx'
predictions = clean_and_predict(file_path)
print(predictions)