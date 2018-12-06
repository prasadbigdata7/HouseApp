import flask
from flask import Flask, render_template, request

import numpy as np
from scipy import misc

# Making Predictions using sk-learn
from sklearn.externals import joblib

# create an instance of the Flask class
app = Flask(__name__)

@app.route('/')
# route the app to index upon entry
@app.route('/index')
def index():
    return flask.render_template('index.html')

# create end point for POST requests (common for form submissions)
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return render_template('index.html', label="No File")
        
        img = misc.imread(file)
        # png often stored with 4 channels, 4th being alpha channel (transparency)
        # since our model has no use for that, take only first 3
        img = img[:,:,:3]
        # row 1, column unknown but will be computed based on the size of the array items
        img = img.reshape(1,-1)

        prediction = model.predict(img)

        label = str(np.squeeze(prediction))

        # SVHN dataset use 10 to represent digit '0'
        if label=='10':
            label='0'
        return render_template('index.html', label=label, file=file)

# initialize the app by loading our ML model and running on localhost and port 8181
if __name__ == '__main__':
    model = joblib.load('models/rfClassifier.pkl')
    app.run(host='0.0.0.0', port=8181, debug=True)