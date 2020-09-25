from flask import Flask,render_template,request
#--------

import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import random
import json

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
stemmer = PorterStemmer()

nltk.download('punkt')
#--------

# Flask
app = Flask(__name__)



@app.route('/')
def home():

    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    #Get user input
    sentence = request.form['rate']
   
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    #########
    print(prob,tag)
    ########

    # response
    if prob.item() > 0.20:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return render_template('index.html',prediction_text=random.choice(intent['responses']))
                
    else:
        print("I do not understand...")
        return render_template('index.html', prediction_text="I do not understand...")


if __name__ == '__main__':
    app.debug = True
    app.run() 