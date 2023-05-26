import json
import torch
import pickle
import numpy as np
from flask import Flask, request, render_template
import joblib
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="root",
  database="restaurant"
)
mycursor = mydb.cursor()
app = Flask(__name__)

# Load the model and tokenizer
model_path = 'bert_sentiment_model.pkl'
tokenizer_path = 'bert_tokenizer.pkl'

model = torch.load(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/')

def home():
    return render_template('index.html')


@app.route('/restaurant', methods=['GET'])
def restaurant():
    query = 'SELECT Name FROM resto;'
    mycursor.execute(query)
    row_headers = [x[0] for x in mycursor.description]  # this will extract row headers
    rv = mycursor.fetchall()
    json_data = []
    for result in rv:
        json_data.append(dict(zip(row_headers, result)))
    print(json_data)
    return json.dumps(json_data)


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        review = request.form['Review']
        name = request.form['restaurant']
        data = {'review': review}
        json_data = json.dumps(data)
        inputs = tokenizer.encode_plus(
            json_data,
            None,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
        sentiment = 'positive' if predictions[1] > predictions[0] else 'negative'
        if sentiment == 'positive':
            prediction = True
        else:
            prediction = False

    if prediction:
        query = 'INSERT INTO `restaurant`.`rating` (`Name`,`Rating`,`Tweet`,`Sentiment`) VALUES(\'{}\', {}, \'{}\', {});'.format(name, 5, review, 1)
        mycursor.execute(query)
        mydb.commit()
        query = 'UPDATE resto SET Previous_Rating = Current_Rating, Current_Rating = (SELECT AVG(Rating) FROM rating WHERE Name=\'{}\') WHERE Name=\'{}\''.format(name, name)
        mycursor.execute(query)
        mydb.commit()
        getOldNewRating(name)
        response = getOldNewRating(name)
        return render_template('predict.html', response=response, review='Positive')
    else:
        query = 'INSERT INTO `restaurant`.`rating` (`Name`,`Rating`,`Tweet`,`Sentiment`) VALUES(\'{}\', {}, \'{}\', {});'.format(
            name, 1, review, 0)
        mycursor.execute(query)
        mydb.commit()
        query = 'UPDATE resto SET Previous_Rating = Current_Rating, Current_Rating = (SELECT AVG(Rating) FROM rating WHERE Name=\'{}\') WHERE Name=\'{}\''.format(
            name, name)
        mycursor.execute(query)
        mydb.commit()
        response = getOldNewRating(name)
        return render_template('predict.html', response=response, review='Negative')

def getOldNewRating(name:str):
    query = 'SELECT CONVERT(Previous_Rating, CHAR) PR, CONVERT(Current_Rating,CHAR) CR FROM resto WHERE Name = \'{}\';'.format(name)
    mycursor.execute(query)
    row_headers = [x[0] for x in mycursor.description]  # this will extract row headers
    rv = mycursor.fetchall()
    json_data = []
    for result in rv:
        json_data.append(dict(zip(row_headers, result)))
    json_data[0]["Name"] = name
    print(json_data)
    return json_data[0]

if __name__ == "__main__":
    app.run(debug=True)
