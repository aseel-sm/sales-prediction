import pickle
from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np
import joblib
import xgboost as xgb
import pandas as pd
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def result():
    print(request.data)
    item_weight= float(request.form['weight'])
    item_fat_content=float(request.form['item_fat_content'])
    item_visibility= float(request.form['visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['mrp'])
    outlet_establishment_year= float(request.form['outlet_establishment_year'])
    outlet_size= float(request.form['outlet_size'])
    outlet_location_type= float(request.form['outlet_location_type'])
    outlet_type= float(request.form['outlet_type'])

    X= np.array([[ item_weight,item_fat_content,item_visibility,item_type,item_mrp,outlet_establishment_year,
                  outlet_size,outlet_location_type,outlet_type ]])
    X=pd.DataFrame(X,columns=['item_weight', 'item_fat_content', 'item_visibility', 'item_type',
       'item_mrp', 'outlet_establishment_year', 'outlet_size',
       'outlet_location_type', 'outlet_type'])
    print(X)

    

    model_path1=r'xgb.sav'
    model_path2=r'cat.sav'
    model_path3=r'lgb.sav'

    # model1= joblib.load(open(model_path1, 'rb'))
    model2= joblib.load(open(model_path2, 'rb'))
    model3= joblib.load(open(model_path3, 'rb'))
    model1=xgb.Booster()
    model1.load_model('xgb.json')
    # pred_1=model1.predict(X)
    pred_2=model2.predict(X)
    pred_3=model3.predict(X)
    print(pred_2,pred_3)
    return jsonify({'pred':{'xgb':round(float(0),3),'cat': round(float(pred_2),3),
    'lbg':round(float(pred_3),3)}})

if __name__ == "__main__":
    app.run(debug=True, port=9457)
