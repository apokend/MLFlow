from flask import Flask, request, jsonify
from argparse import ArgumentParser
import mlflow
from model import CnnSentimentAnalysis
from utils.helpers import *
from collections import Counter
from string import punctuation
import torch
import os

app = Flask(__name__)

# parser = ArgumentParser()
# parser.add_argument('--model', type=str, help='Model uri')
# args=parser.parse_args()


@app.route('/', methods=['POST','GET'])
def test():
    #model = mlflow.pytorch.load_model(args.model)
    model = mlflow.pytorch.load_model(os.environ['SPARK_HOME'])
    model = model.to('cpu')
    data = request.get_json()
    msg = [data['msg']]

    all_reviews, words=review_formatting(msg)
    count_words = Counter(words)
    total_words=len(words)
    sorted_words=count_words.most_common(total_words)
    vocab_to_int={w:i+1 for i,(w,c) in enumerate(sorted_words)}

    features=preprocess(msg, vocab_to_int)
    features = torch.from_numpy(features)

    output = torch.sigmoid(model(features))
    print(output)
    return jsonify({'output':f'{output}'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001')
