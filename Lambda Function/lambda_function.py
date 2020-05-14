"""
Copyright 2020, Sai Bhargav Dasari, All rights reserved
"""
import boto3
import json
from functools import reduce
from pomegranate import HiddenMarkovModel


s3 = boto3.client('s3')


training_set_vocab = []
with open('training_vocab.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        current_item = line[:-1]
        training_set_vocab.append(current_item)


def replace_unknown(sequence):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in training_set_vocab else 'nan' for w in sequence]


def simplify_decoding(X, model):
    """X should be a 1-D sequence of observations for the model to predict"""
    _, state_path = model.viterbi(replace_unknown(X))
    # do not show the start/end state predictions
    decodings = [state[1].name for state in state_path[1:-1]]
    string_out = [x+'--->'+y for x, y in zip(X, decodings)]
    return(reduce(lambda x, y: x+" "+y, string_out))


def lambda_handler(event, context):
    # TODO implement
    content_object = s3.get_object(Bucket='bhargav-ml-trained-models', Key='pos_model.txt')
    file_content = content_object['Body'].read().decode()
    json_content = json.loads(file_content)
    model = HiddenMarkovModel.from_json(json_content)
    sentence = event['body'].split(' ')
    output = simplify_decoding(sentence, model)
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/plain', 'Access-Control-Allow-Origin': '*'},
        'body': output
    }
