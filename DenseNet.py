#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:09:46 2022

@author: fabianodicheti
"""
import os
import timeit
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

tf.debugging.set_log_device_placement(True)



def bloco_denso(x, filtros):
    for i in range(4):
        y = tf.keras.layers.Conv2D(filtros, 3, padding='same', activation='relu')(x)
        y = tf.keras.layers.BatchNormalization()(y)
        x = tf.keras.layers.concatenate([x,y])
    return x
    
def bloco_transicao(x, filtros):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filtros, 1, activation='relu')(x)
    x = tf.keras.layers.AvgPool2D(2)(x)
    return x
    


def DenseNet(uniq, check_point: str = 'NULL'):
    '''arquitetura da rede neural convolucional'''
    with tf.device('/gpu:0'):
        datagen_treino = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        
        training_set = datagen_treino.flow_from_directory(
            'dataset/training_set',
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary')
        
        datagen_teste = ImageDataGenerator(rescale=1./255)
        
        test_set = datagen_teste.flow_from_directory(
            'dataset/test_set',
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary')
        
        #model
        
        inputs = tf.keras.Input(shape=(224,224,3))
        
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
        
        x = tf.keras.layers.MaxPool2D(2)(x)
        
        x = bloco_transicao(x, 64)
        
        for i in range(4):
            x = bloco_denso(x, 64)
            x = bloco_transicao(x, 128)
            
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
                      
        '''              
        es_callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        patience = 4,
        min_delta = 0.002,
        restore_best_weights = True)
        
        
        
        class_weights = {
            0: 1.0,  # peso da classe 0
            1: 1.0,  # peso da classe 1
            }
        '''
        
        start = timeit.default_timer()
        
        model.fit(
            x=training_set,
            validation_data=test_set,
            epochs=20 #,callbacks = [es_callback]
            )

        stop = timeit.default_timer()
        cronometro = str(stop - start)

        #model.save('dataset/'+uniq +'DN_model')
        
        print('time :  ', cronometro)



aleatorio = str(np.random.randint(1000,10000))
DenseNet(aleatorio)

