import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
# Load the datasets
uniprot_data = pd.read_excel("uploads/UniProt_Data.xlsx", header=1)
protein_name = pd.read_excel("uploads/ProtienName.xlsx", header=1)

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
    df = pd.read_excel(file_path)
    #df['Your_Column'] = df['Your_Column'].astype(str).str[:4]
    # Perform PCA and identify top 10 proteins
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.iloc[:, 1:])
    pca = PCA(n_components=10)
    pca.fit(scaled_data)
    feature_importances = pca.components_
    protein_names = df.columns[1:]
    top_proteins = pd.Series(feature_importances[0], index=protein_names).abs().sort_values(ascending=False).head(10)
    top_proteins_names = top_proteins.index.tolist()

    # Map OIDs to UniProt names and then to Protein Names
    olink_to_uniprot_map = uniprot_data.set_index('OlinkID')['UniProt'].dropna().to_dict()
    uniprot_to_protein_map = protein_name.set_index('Uniprot')['Protein target'].dropna().to_dict()
    top_proteins_protein_names = [uniprot_to_protein_map.get(olink_to_uniprot_map.get(oid)) for oid in top_proteins_names]

    # Replace OIDs with Protein Names in the DataFrame used for linear regression
    protein_replace_dict = {oid: protein_name.strip() for oid, protein_name in zip(top_proteins_names, top_proteins_protein_names) if protein_name}

    # Creating a new DataFrame to ensure no unwanted columns are included
    df_renamed = df.rename(columns=protein_replace_dict)
    features_df = df_renamed.loc[:, df_renamed.columns.str.contains('D0|D3')]
    target_df = df_renamed.loc[:, df_renamed.columns.str.contains('D7')]

    # Select only the columns for the top proteins for D7
    target_df = target_df[[col for col in target_df.columns if col.split('_')[0] in protein_replace_dict.values()]]

    # Now pivot the DataFrame to get the features and target for regression
    # Here, we extract the 'Public_ID' and ensure that we copy the data to avoid SettingWithCopyWarning
    filtered_df = df_renamed[['Public_ID'] + list(protein_replace_dict.values())].copy()
    filtered_df[['UserID', 'Day']] = filtered_df['Public_ID'].str.split('_', expand=True)
    filtered_df = filtered_df.drop('Public_ID', axis=1)
    pivoted_df = pd.pivot_table(filtered_df, index='UserID', columns='Day', aggfunc='first')

    # Flatten the MultiIndex in columns after pivoting
    pivoted_df.columns = [f'{protein}_{day}' for protein, day in pivoted_df.columns]
    # Separate features and target for regression
    X = pivoted_df[[col for col in pivoted_df.columns if '_D0' in col or '_D3' in col]].fillna(pivoted_df.mean())
    y = pivoted_df[[col for col in pivoted_df.columns if '_D7' in col]].fillna(pivoted_df.mean())

    # Train linear regression model
    model = LinearRegression()
    predictions = pd.DataFrame(index=y.index)

    # Train a model for each protein's D7 values
    for protein in protein_replace_dict.values():
        protein_features = [col for col in X if col.startswith(protein)]
        protein_target = f'{protein}_D7'
        if protein_target in y.columns:
            model.fit(X[protein_features], y[protein_target])
            predictions[protein] = model.predict(X[protein_features])

    # Save the D7 predictions to an Excel file with protein names
    predictions_excel_file_path = 'output.xlsx'
    
    predictions.to_excel(predictions_excel_file_path)
    return predictions_excel_file_path

if __name__ == '__main__':
    app.run(debug=True)
