from flask import Flask,request, render_template
import numpy as np
import pickle
import sklearn
#loading models
model = pickle.load(open('model.pkl','rb'))
#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        
        Item_MPR = request.form["Item_MPR"]
        Outlet_type = request.form["Outlet_type"]
        Outlet_identifier = request.form["Outlet_identifier"]
        Outlet_size = request.form["Outlet_size"]
        Item_visibility = request.form["Item_visibility"]
        Outlet_location_type = request.form["Outlet_location_type"]
        Outlet_established_year = request.form["Outlet_established_year"]
        
        
        


        features = np.array([[Item_MPR,Outlet_type,Outlet_identifier,Outlet_size,Item_visibility,Outlet_location_type,Outlet_established_year]],dtype=np.float32)
        transformed_feature = features.reshape(1, -1)
        prediction = model.predict(transformed_feature)[0]
        print(prediction)
        return render_template('index.html',prediction = prediction)

if __name__=="__main__":
    app.run(debug=True)