import pickle
from flask import Flask,app,request,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
# import pickle

# with open('gbregression.pkl', 'rb') as f:
    # model,scaler = pickle.load(f)
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.simplefilter("error", InconsistentVersionWarning)

app=Flask(__name__)
model,scaler=pickle.load(open('gbregression.pkl','rb'))


# model=pickle.load(open("gbregression.pkl",'rb'))

# regmodel=pickle.load(open('gbregression.pkl','rb'))




@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route('/predict',methods=['Post'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text=f"The PRedicted House price is {output} ")
    
if __name__=="__main__":
    app.run(debug=True)