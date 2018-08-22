# -*- coding: utf-8 -*-
import os

import pandas as pd
from flask import Flask, render_template
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from sklearn.externals import joblib
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow as tf

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, u'Image only!'), FileRequired(u'File was empty!')])
    submit = SubmitField(u'Upload')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)
        create_graph(model_path)
        feature = extract_features(filename, False)
        imf= prediction_imf(feature)
        ms=prediction_ms(feature)
    else:
        filename =''
        file_url = None
        ms = ''
        imf = 0.0
    return render_template('index.html', form=form, file_url=file_url, ms=ms, imf=imf, fileName=filename)


model_path = '/home/arannya/TRABEYA/Data_conference/NASNet-master/imagenet/tensorflow_inception_graph.pb'


def create_graph(model_path):
    """
    create_graph loads the inception model to memory, should be called before
    calling extract_features.

    model_path: path to inception model in protobuf form.
    """
    with gfile.FastGFile(model_path, 'rb') as f:
        print(f)
        graph_def = tf.GraphDef()
        print(graph_def)
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def extract_features(image_paths, verbose=False):
    print(image_paths)
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))

    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        # for i, image_path in enumerate(image_paths):
        if verbose:
            print('Processing %s...' % image_paths)

        if not gfile.Exists(image_paths):
            tf.logging.fatal('File does not exist %s', image_paths)

        image_data = gfile.FastGFile(image_paths, 'rb').read()

        feature = sess.run(flattened_tensor, {
            'DecodeJpeg/contents:0': image_data
        })

        features[0, :] = np.squeeze(feature)

    # print(len(features), features[0])
    return features


def prediction_imf(feature):
    pipe = joblib.load('trained_model_imf.sav')
    print('----------------', pipe)

    pred = pd.Series(pipe.predict(feature))
    print(pred.tolist()[0])
    return pred.tolist()[0]


def prediction_ms(feature):
    pipe = joblib.load('svm_model.sav')
    print('----------------', pipe)

    pred = pd.Series(pipe.predict(feature))
    print(pred.tolist()[0])
    return pred.tolist()[0]


if __name__ == '__main__':
    app.run()
