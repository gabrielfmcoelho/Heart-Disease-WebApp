import numpy as np
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load('models/model.joblib')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/quest_form')
def quest_form():
    return render_template('quest_form.html')

@app.route('/result',methods=['POST'])
def result():
    #int_features = [str(x) for x in request.form.values()]
    features = [np.array(request.form.values())] 
    prediction = model.predict(features) 
    result = prediction[0]

    if result == 1:
        prediction_text = 'Há alta suspeita de doença cardiaca'
    else:
        prediction_text = 'Há baixa suspeita de doença cardiaca'
    
    return render_template('result.html', prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)