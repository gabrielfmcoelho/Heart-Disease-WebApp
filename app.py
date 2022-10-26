import pandas as pd
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
    list_cat_features = ['Sex',
           'ChestPainType',
           'RestingECG',
           'ST_Slope']
    list_num_features = ['Age',
           'RestingBP',
           'Cholesterol',
           'FastingBS',
           'MaxHR',
           'Oldpeak',
           'ExerciseAngina']
    list_features = list_cat_features + list_num_features

    cat_features = []
    num_features = []

    for x in list_cat_features:
        valor_cat = str(request.form.get(x))
        cat_features.append(valor_cat)

    for x in list_num_features:
        valor_num = int(request.form.get(x))
        num_features.append(valor_num)

    features_unidas = cat_features + num_features
    df = pd.DataFrame([features_unidas], columns=list_features)
    prediction = model.predict(df) 
    result = prediction[0]

    print(df)
    print(result)

    return render_template('result.html', previsao=result)


if __name__ == "__main__":
    app.run(debug=True)