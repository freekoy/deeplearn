from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout,Reshape
from keras.optimizers import RMSprop
from keras import backend as K
import keras


# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'tigerandpanda/train'
validation_data_dir = 'tigerandpanda/test'
epochs = 30 #训练批次
batch_size = 32 #训练量
num_classes = 2 #类别

model = Sequential()
model.add(Reshape((49152,), input_shape=(128,128,3)))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

#从文件夹里读取出图片 并转换格式 图片格式为两种 （28，28，1）灰度图 和 (28, 28, 3)彩色图
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

# model.save('zoo_mlp.h5')#保存模型
