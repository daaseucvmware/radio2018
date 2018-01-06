# Make an API

# Load the libraries
import numpy as np
from sklearn.externals import joblib

#Load the model
model = joblib.load('simple-model.pkl')

#Create a predict function
def predict(hourOfTheDay, memoryUsage):
    features= [[hourOfTheDay, memoryUsage]]
    prob = model.predict(features)[0]
    return prob[:0]