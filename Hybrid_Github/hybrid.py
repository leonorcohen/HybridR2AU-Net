import tensorflow as tf


from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, add, multiply
from tensorflow.keras.layers import concatenate, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K

from evaluation import *

def up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])

    return concate

def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x

def attention_up_and_concate(down_layer, layer, data_format='channels_last'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    # up = Conv2DTranspose(out_channel, [2, 2], strides=[2, 2])(down_layer)
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate



def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],
              padding='same', data_format='channels_first'):
    
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    layer = input_layer
    for i in range(2):
        layer = Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
        layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    out_layer = add([layer, skip_layer])
    return out_layer


# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1], 
                  padding='same', data_format='channels_first'):
    
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(3):
            if i == 0:

                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer



def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x




def hybrid(pretrained_weights = None, input_size = (512,640,1), n_label = 1, data_format='channels_last'):
    inputs = Input(input_size)
    x = inputs
    depth = 4
    features = 32
    skips = []
    
    # Encoder
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    # RecRes Block 5
    x = rec_res_block(x, features, data_format=data_format)
    
    features = features // 2
    up32 = UpSampling2D(size = (2,2))(skips[3])
    up23 = UpSampling2D(size = (2,2))(up32)
    up14 = UpSampling2D(size = (2,2))(up23)
    
    # RecRes Block 6 
    x = attention_up_and_concate(x, skips[3], data_format=data_format)
    x = rec_res_block(x, features, data_format=data_format)
    
    features = features // 2
    up22 = UpSampling2D(size = (2,2))(skips[2])
    up13 = UpSampling2D(size = (2,2))(up22)
    
    # Dense Layer 32
    skip = concatenate([skips[2], up32], axis=3)
    skip = conv2d_bn(skip, features, 3, 3, activation='relu', padding='same')
    
    # RecRes Block 7
    x = attention_up_and_concate(x, skip, data_format=data_format)
    x = rec_res_block(x, features, data_format=data_format)
    
    features = features // 2
    up12 = UpSampling2D(size = (2,2))(skips[1])
    
    # Dense Layer 22
    skip = concatenate([skips[1], up22], axis=3)
    skip = conv2d_bn(skip, features, 3, 3, activation='relu', padding='same')
    # Dense Layer 23
    skip = concatenate([skip, up23], axis=3)
    skip = conv2d_bn(skip, features, 3, 3, activation='relu', padding='same')
    
    # RecRes Block 8
    x = attention_up_and_concate(x, skip, data_format=data_format)
    x = rec_res_block(x, features, data_format=data_format)

    features = features // 2
    
    # Dense Layer 12
    skip = concatenate([skips[0], up12], axis=3)
    skip = conv2d_bn(skip, features, 3, 3, activation='relu', padding='same')
    # Dense Layer 13
    skip = concatenate([skip, up13], axis=3)
    skip = conv2d_bn(skip, features, 3, 3, activation='relu', padding='same')
    # Dense Layer 14
    skip = concatenate([skip, up14], axis=3)
    skip = conv2d_bn(skip, features, 3, 3, activation='relu', padding='same')
    
    # RecRes Block 9
    x = attention_up_and_concate(x, skip, data_format=data_format)
    x = rec_res_block(x, features, data_format=data_format)
    
    # Last Convolution Layer
    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=conv6)

    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy', dice_coef, jacard])
    model.summary()
    
    return model
