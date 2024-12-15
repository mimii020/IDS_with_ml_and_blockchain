from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)

with open('IDS_model.pkl', 'rb') as file:
    model = pickle.load(file)

def label_encode(df):
  for col in df.columns:
    if df[col].dtype == 'object':
      label_encoder = LabelEncoder()
      df[col] = label_encoder.fit_transform(df[col])

def preprocess_input(data):
    input_df = pd.DataFrame([data])
    
    #handling missing values
    input_df.fillna(0)
    #label encoding
    label_encode(input_df)
    


@app.route('/predict', methods=['POST'])
def predict(): 
    try:
      input_data = request.get_json()
      processed_data = preprocess_input(input_data)
      prediction = model.predic(processed_data)

      response = {'predicion': prediction.toList()}
      return jsonify(response)
    except Exception as e:
      return jsonify({'error': str(e)})
    
if __name__ == '__main__':
   app.run(debug=True)