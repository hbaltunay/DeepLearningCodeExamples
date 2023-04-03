import math
import numpy as np
import tensorflow as tf
from keras.layers import IntegerLookup
from keras.layers import Normalization
from keras.layers import StringLookup

class functions:

    def dataframe_to_dataset(dataframe, target, batch_size):

        dataframe = dataframe.copy()
        labels = dataframe.pop(target)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        ds = ds.shuffle(buffer_size=len(ds))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)
        return ds
    
    def input_Model(features, dtype):
        
        inputs = list()
            
        for feature in features:
            i = tf.keras.Input(shape=(1,), name=feature, dtype=dtype)
            inputs.append(i)
            
        return inputs
        
    def encode_numerical_feature(feature, name, dataset):
            
        normalizer = Normalization()

        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

        normalizer.adapt(feature_ds)

        encoded_feature = normalizer(feature)

        return encoded_feature


    def encode_categorical_feature(feature, name, dataset, is_string, use_embedding=None, vocabulary=None):
        
        lookup_class = StringLookup if is_string else IntegerLookup
        
        if vocabulary:

            lookup = lookup_class(
                    vocabulary=vocabulary,
                    output_mode="int" if use_embedding else "binary",
                )

        else:

            lookup = lookup_class("int" if use_embedding else "binary")

            feature_ds = dataset.map(lambda x, y: x[name])
            feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

            lookup.adapt(feature_ds)

            vocabulary = lookup.get_vocabulary()

        
        if use_embedding:

            encoded_feature = lookup(feature) 

            vocabulary = lookup.get_vocabulary()

            embedding_dims = int(math.sqrt(len(vocabulary)))

            embedding = tf.keras.layers.Embedding(
                input_dim=len(vocabulary), output_dim=embedding_dims
            )

            encoded_feature = embedding(encoded_feature)
            
            encoded_feature = tf.keras.layers.Reshape((encoded_feature.shape[-1],))(encoded_feature)

        else:
            
            encoded_feature = lookup(feature)

        return encoded_feature
    
    @classmethod
    def features_Model(self, features, name, dataset, encode, is_string=None, use_embedding=None, vocabulary=None):
        inputs = list()
        
        if encode == "numerical":
            for index, feature in enumerate(features):
                inputs.append(self.encode_numerical_feature(feature, name[index], dataset))
        elif encode == "categorical":
            for index, feature in enumerate(features):
                inputs.append(self.encode_categorical_feature(feature, name[index], dataset, is_string, use_embedding, vocabulary[index]))
            
        return inputs