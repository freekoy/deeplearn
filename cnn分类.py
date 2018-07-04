# encoding: utf-8

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
# from keras.optimizers import RMSprop
from keras import backend as K
import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'tigerandpanda/train'
validation_data_dir = 'tigerandpanda/test'
epochs = 20 #训练批次
batch_size = 32 #训练量
num_classes = 2 #类别

# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

#从文件夹里读取出图片 并转换格式
train_generator = train_datagen.flow_from_directory(
    train_data_dir,#文件夹
    target_size=(img_width, img_height),#裁剪图片尺寸为某值
    # color_mode='grayscale',#灰度图 选项
    batch_size=batch_size,#生成训练量
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    # color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    generator=train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    )

# model.save('zoo_cnn.h5')#保存模型
