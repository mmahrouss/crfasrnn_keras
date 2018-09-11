"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add
from crfrnn_layer import CrfRnnLayer
import tensorflow as tf


def get_crfrnn_model_def():
    """ Returns Keras CRN-RNN model definition.

    Currently, only 500 x 500 images are supported. However, one can get this to
    work with different image sizes by adjusting the parameters of the Cropping2D layers
    below.
    """

    channels, height, weight = 3, 500, 500

    # Input
    input_shape = (height, weight, 3)
    img_input = Input(shape=input_shape)

    # Add plenty of zero padding
    x = ZeroPadding2D(padding=(100, 100))(img_input)

    # VGG-16 convolution block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='conv1_1',trainable=False)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2',trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # VGG-16 convolution block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1', trainable=False)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2', padding='same', trainable=False)(x)

    # VGG-16 convolution block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2', trainable=False)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3', padding='same', trainable=False)(x)
    pool3 = x

    # VGG-16 convolution block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4', padding='same', trainable=False)(x)
    pool4 = x

    # VGG-16 convolution block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2', trainable=False)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3', trainable=False)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5', padding='same', trainable=False)(x)

    # Fully-connected layers converted to convolution layers
    x = Conv2D(4096, (7, 7), activation='relu', padding='valid', name='fc6', trainable=False)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='valid', name='fc7', trainable=True)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(21, (1, 1), padding='valid', name='score-fr', trainable=True)(x)

    # Deconvolution
    score2 = Conv2DTranspose(21, (4, 4), strides=2, name='score2', trainable=True)(x)

    # Skip connections from pool4
    score_pool4 = Conv2D(21, (1, 1), name='score-pool4', trainable=True)(pool4)
    score_pool4c = Cropping2D((5, 5))(score_pool4)
    score_fused = Add()([score2, score_pool4c])
    score4 = Conv2DTranspose(21, (4, 4), strides=2, name='score4', use_bias=False, trainable=True)(score_fused)

    # Skip connections from pool3
    score_pool3 = Conv2D(21, (1, 1), name='score-pool3', trainable=True)(pool3)
    score_pool3c = Cropping2D((9, 9))(score_pool3)

    # Fuse things together
    score_final = Add()([score4, score_pool3c])

    # Final up-sampling and cropping
    upsample = Conv2DTranspose(21, (16, 16), strides=8, name='upsample', use_bias=False, trainable=True)(score_final)
    upscore = Cropping2D(((31, 37), (31, 37)))(upsample)
    print(upscore)
    output = CrfRnnLayer(image_dims=(height, weight),
                         num_classes=21,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([upscore, img_input])
    output = keras.layers.Reshape((250000,21),name='Reshape')(output)
    
    # Build the model
    model = Model(img_input, output, name='crfrnn_net')

    return model
