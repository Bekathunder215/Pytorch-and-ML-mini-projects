from flask import Flask, jsonify, request
from utilities import predict, CNN
import json

app = Flask(__name__)

@app.route("/")
def home():
    # print(predict())
    return "You are at HOME"

@app.route("/predict")
def prediction():
    try:
        data = json.loads(predict())
    except KeyError:
        return jsonify({"error" : "there was an error"})
    return data


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
