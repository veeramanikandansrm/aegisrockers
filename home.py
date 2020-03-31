from flask import Flask, url_for
from distutils.util import strtobool
import numpy as np
app = Flask(__name__)
from joblib import dump, load
model = load('model_xg.joblib') 
@app.route('/')
def api_root():
    return 'Welcome'

@app.route('/predict/<int:store>/<int:item>/<int:day>/<int:month>/<int:year>')
def api_predict(store,item,day,month,year):
	x=np.array([[store,item,day,month,year]])
	z=model.predict(x)
	return 'Prediction:'+ str(z[0]) 
if __name__ == '__main__':
    app.run()
