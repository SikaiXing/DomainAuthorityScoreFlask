from flask import jsonify, render_template, request

import requests

from FlaskWebProject2 import app

from . import tokenization

import json
import os
import csv

import tensorflow as tf
import collections

import argparse

import time



import grpc

from tensorflow.contrib.util import make_tensor_proto



from tensorflow_serving.apis import predict_pb2

from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.core.framework import types_pb2

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, market, domain, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a  # Used as query
        self.text_b = text_b
        self.label = label
        self.market = market
        self.domain = domain

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 guid,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 market_id,
                 domain_id,
                 is_real_example=True):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.market_id = market_id
        self.domain_id = domain_id
        self.is_real_example = is_real_example


def get_map_from_file(file):
    map = {}
    with tf.gfile.Open(file, "r") as f:
        reader = csv.reader(f)
        index = 0
        for line in reader:
            if len(line) != 0 and line != "":
                map[line[0]] = index
                index += 1
        return map


def GetDomain(url):
    url = url.lower()
    domain = ""
    tmp = ""

    if url.startswith("http://"):
        tmp = url[7:]
    elif url.startswith("https://"):
        tmp = url[8:]
    elif url.startswith("www."):
        tmp = url
    else:
        tmp = url
    
    portIndex = tmp.find(':')
    if portIndex > 0:
        tmp = tmp[0:portIndex]
    index = tmp.find('/')
    if index > 0:
        domain = tmp[0:index]
    else:
        domain = tmp
    return domain


label_list = ["0", "1"]
tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.realpath(
    os.path.join(os.getcwd(), "vocab.txt")), do_lower_case=True)

market_map = get_map_from_file("market_dic.tsv")
domain_map = get_map_from_file("domain_dic.tsv")

@app.route('/', defaults={'js': 'plain'})
@app.route('/<any(plain):js>')
def index(js):
    return render_template('{0}.html'.format(js), js=js)


@app.route('/calculate', methods=['POST'])
def calculate():
    query = request.form.get("query", "", type=str)
    market = request.form.get("market", "", type=str)
    domain = request.form.get("domain", "", type=str)
    url = request.form.get("url", "", type=str)

    if len(domain.strip())==0 and len(url.strip())==0:
        return jsonify(result="url and domain cannot both be empty!")
    elif len(url.strip())!=0:
        domain = GetDomain(url)
    else:
        domain = domain
    
    guid = "predict0"
    query = tokenization.convert_to_unicode(query)
    market = tokenization.convert_to_unicode(market)
    domain = tokenization.convert_to_unicode(domain)
    example = InputExample(guid=guid, text_a=query, market=market, domain=domain, label="1")
    tf_example = convert_single_example(example, label_list, market_map, domain_map, 16, tokenizer)

    channel = grpc.insecure_channel('localhost:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    start = time.time()
    predict_request = predict_pb2.PredictRequest()
    predict_request.model_spec.name = "domainauthority"
    predict_request.model_spec.signature_name = "predict"
    predict_request.inputs['examples'].CopyFrom(
        make_tensor_proto([tf_example.SerializeToString()], shape=[1], dtype=types_pb2.DT_STRING))
    result = stub.Predict(predict_request, 10.0)
    end = time.time()
    prediction = result.outputs['output'].float_val[0]

    return jsonify(result=str(prediction))


def convert_single_example(example, label_list, market_map, domain_map, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if example.market in market_map:
        market_id = market_map[example.market]
    else:
        market_id = 0
    if example.domain in domain_map:
        domain_id = domain_map[example.domain]
    else:
        domain_id = 0

    feature = InputFeatures(
        guid=example.guid,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        market_id=market_id,
        domain_id=domain_id,
        is_real_example=True)
    features = collections.OrderedDict()
    features["guid"] = create_bytes_feature(feature.guid)
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["market_ids"] = create_int_feature([feature.market_id])
    features["domain_ids"] = create_int_feature([feature.domain_id])
    features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

def create_bytes_feature(value):
    f = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
    return f

