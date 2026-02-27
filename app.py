from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
 
model = pickle.load(open("profit_model.pkl", "rb"))
columns = pickle.load(open("profit_columns.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    
    sales = float(request.form['Sales'])
    quantity = float(request.form['Quantity'])
    discount = float(request.form['Discount'])
    postal = float(request.form['Postal Code'])
 
    input_data = pd.DataFrame([{
        'Sales': sales,
        'Quantity': quantity,
        'Discount': discount,
        'Postal Code': postal
    }])

     
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[columns]

    
    prediction = model.predict(input_data)[0]

    return render_template("index.html",
                           prediction_text="Predicted Profit: " + str(round(prediction, 2)))

if __name__ == "__main__":
    app.run(debug=True)