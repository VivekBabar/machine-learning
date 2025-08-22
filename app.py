from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from model import jarvis_model
import json
from datetime import datetime
import os

app = Flask(__name__)

# Global variable to store the loaded data
df = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        # Get response from the model
        response = jarvis_model.predict(message)
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': f'Error processing message: {str(e)}'})

@app.route('/enhanced_chat', methods=['POST'])
def enhanced_chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        # Get response from the enhanced model
        response = jarvis_model.enhanced_predict(message)
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': f'Error processing message: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    try:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Read the CSV file
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            response = {
                'message': 'File uploaded successfully',
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': list(df.columns),
                'preview': df.head(5).to_dict(orient='records')
            }
            return jsonify(response)
        else:
            return jsonify({'error': 'Please upload a CSV file'})
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'})

@app.route('/analyze', methods=['POST'])
def analyze_data():
    global df
    if df is None:
        return jsonify({'error': 'No data available. Please upload a CSV file first.'})
    
    try:
        data = request.get_json()
        query = data.get('query', '').lower()
        
        response = {'message': '', 'chart_data': None}
        
        if 'summary' in query:
            # Generate summary statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                summary = df[numeric_cols].describe().to_dict()
                
                response['message'] = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
                response['message'] += f"Numeric columns: {', '.join(numeric_cols)}. "
                response['summary'] = summary
            else:
                response['message'] = "No numeric columns found for summary statistics."
                
        elif 'correlation' in query:
            # Calculate correlations
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                # Find the strongest correlation
                max_corr = 0
                max_cols = ('', '')
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > max_corr:
                            max_corr = corr_val
                            max_cols = (corr_matrix.columns[i], corr_matrix.columns[j])
                
                if max_corr > 0:
                    response['message'] = f"The strongest correlation is between {max_cols[0]} and {max_cols[1]} (r = {max_corr:.2f})."
                    # Prepare data for chart
                    response['chart_data'] = {
                        'labels': max_cols,
                        'data': [df[max_cols[0]].mean(), df[max_cols[1]].mean()],
                        'type': 'bar',
                        'title': f'Average values of {max_cols[0]} and {max_cols[1]}'
                    }
                else:
                    response['message'] = "No strong correlations found between numeric columns."
            else:
                response['message'] = "Not enough numeric columns for correlation analysis."
                
        elif 'distribution' in query:
            # Show distribution of a numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = numeric_cols[0]
                response['message'] = f"Distribution of {col}: Mean = {df[col].mean():.2f}, "
                response['message'] += f"Min = {df[col].min():.2f}, Max = {df[col].max():.2f}"
                
                # Prepare data for chart
                response['chart_data'] = {
                    'labels': ['Min', 'Mean', 'Max'],
                    'data': [df[col].min(), df[col].mean(), df[col].max()],
                    'type': 'bar',
                    'title': f'Distribution of {col}'
                }
            else:
                response['message'] = "No numeric columns available for distribution analysis."
                
        elif 'predict' in query or 'forecast' in query:
            response['message'] = "I can help with predictions. Please provide more details about what you'd like to predict."
                
        else:
            response['message'] = "I can help with data summary, correlations, distributions, and predictions. Try asking about one of these topics."
            
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Error analyzing data: {str(e)}'})

@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        success, message = jarvis_model.train_model()
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error retraining model: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)