from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle

model=pickle.load(open("model.pkl",'rb'))

# flask app
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict",methods=["POST"])
def predic():
    features = request.form.get('feature')

    features_lst=features.split(',')
    np_features=np.asarray(features_lst,dtype=np.float32)
    pred=model.predict(np_features.reshape(1,-1))

    output=["cancer" if pred[0]==1 else "not cancer"]
    return render_template('index.html',message=output)




if __name__ == "__main__":
    app.run(debug=True)


