import numpy as np
np.random.seed(1)
import tensorflow as tf
tf.set_random_seed(1234)
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
import scipy.sparse as sp

def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((1, 2))(inputs)
    a = Dense(32, activation='softmax')(a)
    a_probs = Permute((1, 2))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def get_model_dna_pro_att(INIT_LR,EPOCHS,shape0,shape1,shape2,shape3,shape4):
    inputs_phage=Input(shape=(shape0,shape1,shape4))
    inputs_host=Input(shape=(shape2,shape3,shape4))  
    phage_conv_layer = Conv2D(filters = 32,kernel_size = (3,3),padding = "same",activation='relu',name='Conv2d_phage')(inputs_phage)
    phage_batch_layer = BatchNormalization()(phage_conv_layer)
    phage_max_pool_layer = MaxPooling2D(pool_size = (3,3), name='Pooling2d_phage')(phage_batch_layer)
    phage_dt=Dropout(0.5,name='dropout1')(phage_max_pool_layer)
    phage_reshape=Reshape((-1,32))(phage_dt)
    att1=attention_3d_block(phage_reshape)
    host_conv_layer = Conv2D(filters = 32,kernel_size = (3,3),padding = "same",activation='relu',name='Conv2d_host')(inputs_host)
    host_batch_layer = BatchNormalization()(host_conv_layer)
    host_max_pool_layer = MaxPooling2D(pool_size = (2,2), name='Pooling2d_host')(host_batch_layer)
    host_dt=Dropout(0.5,name='dropout2')(host_max_pool_layer)
    host_reshape=Reshape((-1,32))(host_dt)
    att2=attention_3d_block(host_reshape)
    merge_layer=Add()([att1,att2])
    fla=Flatten()(merge_layer)
    dense_layer = Dense(64, activation='relu',name='Dense1')(fla)
    bn=BatchNormalization()(dense_layer)
    dt=Dropout(0.5,name='dropout3')(bn)
    preds = Dense(1, activation='sigmoid',name='Dense2')(dt)
    model=Model([inputs_phage,inputs_host],preds)
    opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['acc'])
    return model
