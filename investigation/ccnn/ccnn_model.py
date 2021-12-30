
# based on # https://www.tu-chemnitz.de/etit/proaut/en/research/ccnn.html, https://www.tu-chemnitz.de/etit/proaut/en/team/stefanSchubert.html


import keras
from keras.layers import Conv2D, Cropping2D, Concatenate, ZeroPadding2D


def CConv2D(filters, kernel_size, input_shape = None, strides=(1, 1), activation='linear', padding='valid', kernel_initializer='glorot_uniform', kernel_regularizer=None):
    def CConv2D_inner(x):
        # TODO: input shape?
        input_height = int(x.get_shape()[0])
        input_width = int(x.get_shape()[1])

        if (input_height % strides[0] == 0):
            pad_along_height = max(kernel_size[0] - strides[0], 0)
        else:
            pad_along_height = max(kernel_size[0] - (input_height % strides[0]), 0)
        if (input_width % strides[1] == 0):
            pad_along_width = max(kernel_size[1] - strides[1], 0)
        else:
            pad_along_width = max(kernel_size[1] - (input_width % strides[1]), 0)
        
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        pad_top_cropping = Cropping2D(cropping=((input_height-pad_top, 0), (0, 0)))(x) # top pad removed
        pad_bottom_cropping = Cropping2D(cropping=((0, input_height-pad_bottom), (0, 0)))(x)

        conc = Concatenate(axis=1)([pad_top_cropping, x, pad_bottom_cropping])
        
        if padding != 'valid':
            raise NotImplementedError()

        cconv2d = Conv2D(filters=filters, kernel_size=kernel_size,
                         strides=strides, activation=activation,
                         padding='valid',
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer)(conc)
        return cconv2d

    return CConv2D_inner