from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D

def setup_model():
    model = Graph()
    model.add_input(name='x', input_shape=(13, 19, 19))

    model.add_node(ZeroPadding2D((2, 2)),   name='z1', input='x')
    model.add_node(Convolution2D(128, 5, 5),name='c1', input='z1')
    model.add_node(Activation('relu'),      name='a1', input='c1')

    model.add_node(ZeroPadding2D((1, 1)),   name='z2', input='a1')
    model.add_node(Convolution2D(64, 3, 3), name='c2', input='z2')
    model.add_node(Activation('relu'),      name='a2', input='c2')

    model.add_node(ZeroPadding2D((1, 1)),   name='z3', input='a2')
    model.add_node(Convolution2D(32, 3, 3), name='c3', input='z3')
    model.add_node(Activation('relu'),      name='a3', input='c3')

    model.add_node(ZeroPadding2D((1, 1)),   name='z4', input='a3')
    model.add_node(Convolution2D(8, 3, 3),  name='c4', input='z4')
    model.add_node(Activation('relu'),      name='a4', input='c4')
    model.add_node(Flatten(),               name='f1', input='a4')

    model.add_node(Dense(64),               name='d1', input='f1')
    model.add_node(Activation('relu'),      name='a5', input='d1')
    model.add_node(Dropout(0.5),            name='dropout', input='a5')

    # Final y1 layer
    model.add_node(Dense(1),                name='d2', input='dropout')
    model.add_node(Activation('linear'),    name='a6', input='d2')
    model.add_output(name='y1', input='a6')

    # Final y2 layer
    model.add_node(Dense(1),                name='d3', input='dropout')
    model.add_node(Activation('linear'),    name='a7', input='d3')
    model.add_output(name='y2', input='a7')

    return model
