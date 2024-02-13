
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
    protein_models = {}

    # Looping over each protein to create a model
    for protein in wide_data['Protein ID'].unique():
        protein_data = wide_data[wide_data['Protein ID'] == protein]
        X = protein_data[['D0', 'D3']]
        y = protein_data['D7']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predicting and evaluating the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Storing the model and its evaluation metric
        protein_models[protein] = {'model': model, 'mse': mse, 'predictions': y_pred}

    return protein_models

# Example usage
file_path = 'K:\\TU_BRAUNSCHWEIG\\PYTHON\\Quaranteam\\TRY1\\Cleaned_NPXvalues.xlsx'
models = clean_and_predict(file_path)
for protein, data in models.items():
    print(f"Protein: {protein}, MSE: {data['mse']}")
