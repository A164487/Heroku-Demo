import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Interface.html')

@app.route('/predict',methods=['POST'])
def predict():

    status = request.form['status']
    edu = request.form['edu']
    occ = request.form['occ']
    twocars = request.form['twocars']
    threecars = request.form['threecars']
    ac = request.form['ac']
    laptop = request.form['laptop']
    hp = request.form['hp']
    tv = request.form['tv']
    internet = request.form['internet']
    water = request.form['water']
    ownlq = request.form['ownlq']
    ownership = request.form['ownership']
    person = request.form['person']
    household = request.form['household']
    income = request.form['income']
    input_variables = pd.DataFrame([[status, edu, occ, twocars, threecars, ac, laptop, hp, tv, internet, water, ownlq, ownership, person, household, income]],
                                    columns=['status', 'edu', 'occ', 'twocars', 'threecars','ac', 'laptop', 'hp', 'tv', 'internet', 'water', 'ownlq', 'ownership', 'person', 'household', 'income'],
                                    dtype=float)
    prediction = model.predict(input_variables)[0]

    if int(prediction) == 2: 
        predict ='You are not in B40 category.'
    else: 
        predict ='You are in B40 category.'  

    return render_template("output.html", prediction = predict) 

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)