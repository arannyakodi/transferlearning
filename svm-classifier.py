import os

import numpy as np
import tensorflow as tf
from sklearn import grid_search, cross_validation
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from tensorflow.python.platform import gfile
import pickle


model_path = 'D:\\NASNet\\NASNet-master\\imagenet\\tensorflow_inception_graph.pb'
image_pth = 'D:\\NASNet\\NASNet-master\\labled_images\\'


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


create_graph(model_path)


def extract_features(image_paths, verbose=False):
    """
    extract_features computed the inception bottleneck feature for a list of images

    image_paths: array of image path
    return: 2-d array in the shape of (len(image_paths), 2048)
    """
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))
    lables = []

    # print(features)
    with tf.Session() as sess:
        flattened_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        # print('-------------------', image_paths)
        for i, image_path in enumerate(image_paths):
            print(image_path)
            print('-----', not gfile.Exists(image_pth + image_path))
            if verbose:
                print('Processing %s...' % (image_path))

            if not gfile.Exists(image_pth + image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            # print(image_pth + image_path)
            # print(gfile.Exists(image_pth + image_path))
            image_data = gfile.FastGFile(image_pth + image_path, 'rb').read()
            print(image_data)
            feature = sess.run(flattened_tensor, {
                'DecodeJpeg/contents:0': image_data
            })

            print(feature)
            if image_path[3] == '1' or image_path[3] == '2' or image_path[3] == '3':
                lables.append('1-3')
            else:
                lables.append(image_path[3])
            '''elif image_path[3] == '4' or image_path[3] == '5' or image_path[3] == '6':
                lables.append('4-6')
            elif image_path[3] == '7' or image_path[3] == '8' or image_path[3] == '9':
                lables.append('7-9')'''

            # lables.append(image_path[3])
            features[i, :] = np.squeeze(feature)

    # print(len(features), features[0])
    return features, lables


def list_files(dir):
    r = []
    for files in os.listdir(dir):
        r.append(files)
    # print(len(r), len(set(r)))
    return r


images_list = list_files(image_pth)
# print(images_list)
feature_matrix_and_lables = extract_features(images_list)
feature_matrix = feature_matrix_and_lables[0]
lables_array = feature_matrix_and_lables[1]

model_output_path = 'D:\\NASNet\\NASNet-master\\output'


def train_svm_classifer(features, labels, model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance

    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2)

    param = [
        {
            "kernel": ["linear"],
            "C": [1000]
        }
    ]
    '''
    ,
        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    '''
    # request probability estimation
    svm = SVC(probability=True)

    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = grid_search.GridSearchCV(svm, param,
                                   cv=20, n_jobs=4, verbose=3)

    clf.fit(X_train, y_train)

    if os.path.exists(model_output_path):
        joblib.dump(clf.best_estimator_, model_output_path)
    else:
        print("Cannot save trained svm model to {0}.".format(model_output_path))

    print("\nBest parameters set:")
    print(clf.best_params_)

    y_predict = clf.predict(X_test)

    labels = sorted(list(set(labels)))
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))

    print("\nClassification report:")
    print(classification_report(y_test, y_predict))


train_svm_classifer(feature_matrix, lables_array, model_output_path)
