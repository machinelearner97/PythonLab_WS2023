import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file.filename != '':
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        output_file_path = clean_and_predict(file_path)
        return send_file(output_file_path, as_attachment=True)
    return jsonify({'error': 'No file selected'})

@app.route('/download', methods=['GET'])
def download_file():
    output_file_path = 'output.xlsx'
    return send_file(output_file_path, as_attachment=True)

def clean_and_predict(file_path):
    # Load the data and perform predictions (same as before)
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

    # List to store predictions for each protein
    protein_predictions_list = []

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

        # Storing the predictions in a dataframe
        protein_predictions = pd.DataFrame({
            'Sample ID': protein_data['Sample ID'],
            'Protein ID': protein,
            'D0': protein_data['D0'],
            'D3': protein_data['D3'],
            'Predicted D7': predictions
        })
        protein_predictions_list.append(protein_predictions)

    # Concatenate all predictions into a single DataFrame
    all_predictions = pd.concat(protein_predictions_list)

    # Saving the predictions to a temporary file in binary mode
    output_file_path = 'output.xlsx'
    all_predictions.to_excel(output_file_path, index=False)
    return output_file_path

if __name__ == '__main__':
    app.run(debug=True)
