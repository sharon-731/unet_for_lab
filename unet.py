import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image


root_directory = r'C:\Users\riyas\Downloads\exudates_unet\exudates_unet'
image_directory = os.path.join(root_directory,r'exudate_img_unet')
mask_directory = os.path.join(root_directory,r'exudate_map_unet')
image_walk_data = list(os.walk(image_directory))
images_filenames = [os.path.join(image_walk_data[0][0],i) for i in image_walk_data[0][2]]
masks_filenames = [os.path.join(mask_directory,i[:-4]+'_hard_exudate.png') for i in image_walk_data[0][2]]

x = []
y = []
for i in range(len(images_filenames)):
    x_current = np.array(Image.open(images_filenames[i]))/255
    y_current = np.array(Image.open(masks_filenames[i]))/255
    x.append(x_current)
    y.append(y_current)
    x.append(np.fliplr(x_current))
    y.append(np.fliplr(y_current))
x = np.stack(x)
y = np.stack(y)

# x = np.stack([np.array(Image.open(i)) for i in images_filenames])
# y = np.stack([np.array(Image.open(i)) for i in masks_filenames])
# x = x/255
# y = y/255

input_image_shape = x.shape[1:]

def double_conv_block(x, n_filters):
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   x = layers.concatenate([x, conv_features])
   x = layers.Dropout(0.3)(x)
   x = double_conv_block(x, n_filters)
   return x

def upsample_block(x, conv_features, n_filters):
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   x = layers.concatenate([x, conv_features])
   x = layers.Dropout(0.3)(x)
   x = double_conv_block(x, n_filters)
   return x

def build_unet_model():
   inputs = layers.Input(shape=input_image_shape)

   f1, p1 = downsample_block(inputs, 64)
   f2, p2 = downsample_block(p1, 128)
   f3, p3 = downsample_block(p2, 256)
   f4, p4 = downsample_block(p3, 512)

   bottleneck = double_conv_block(p4, 1024)

   u6 = upsample_block(bottleneck, f4, 512)
   u7 = upsample_block(u6, f3, 256)
   u8 = upsample_block(u7, f2, 128)
   u9 = upsample_block(u8, f1, 64)

   outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)

   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
   return unet_model

model = build_unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")

results = model.fit(x,y,epochs=5)

model.save_weights('model_checkpoint')