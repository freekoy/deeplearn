# encoding: utf-8

#迁移学习

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Dense, Dropout,Reshape,GlobalAveragePooling2D
from keras.optimizers import RMSprop
from keras import backend as K
import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# dimensions of our images.
img_width, img_height = 128 ,128

train_data_dir = 'tigerandpanda/train'
validation_data_dir = 'tigerandpanda/test'
epochs = 10 #训练批次
batch_size = 16 #训练量
num_classes = 2 #类别 虎与熊猫

input_shape = (img_width, img_height, 3)

# 构建不带分类器的预训练模型
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(1024, activation='relu')(x)

# 添加一个分类器，假设我们有18个类
predictions = Dense(num_classes, activation='softmax')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255)

#从文件夹里读取出图片 并转换格式
train_generator = train_datagen.flow_from_directory(
    train_data_dir,#文件夹
    target_size=(img_width, img_height),#裁剪图片尺寸为某值
    # color_mode='grayscale',#灰度图 选项
    batch_size=batch_size,#生成训练量
    class_mode='categorical')

model.fit_generator(
    generator=train_generator,
    epochs=epochs
    )

# model.save('zoo_cnn.h5')#保存模型
