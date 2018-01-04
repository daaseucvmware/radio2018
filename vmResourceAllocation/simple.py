# Make an API

# Load the libraries
import numpy as np
from sklearn.externals import joblib

#Load the model
model = joblib.load('simple-model.pkl')

#Create a predict function
def predict(hourOfTheDay):
    features= [hourOfTheDay]
    print ("%d, %d"  % (model.predict_proba([features])[0]))
    prob1 = model.predict_proba([features])[0]
    return prob1