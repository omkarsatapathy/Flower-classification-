import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
train_data_dir = r'D:\Dataset\flowers\train'
validation_data_dir = r'D:\Dataset\flowers\valid'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest',)
      #validation_split=0.2)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(150,150),
        #batch_size =16,
        class_mode='categorical',
        shuffle=True)
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(150,150),
        #batch_size=16,
        class_mode='categorical',
        shuffle=False)

model = keras.Sequential()
model.add(keras.layers.Conv2D(64,(5,5),padding='Same',activation='relu',input_shape=(150,150,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(128,(3,3),padding='Same',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(128,(3,3),padding='Same',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(96,(3,3),padding='Same',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(keras.layers.Conv2D(128,(3,3),padding='Same',activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(5,activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9),
              metrics = ['accuracy'])
my_calls = [keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=2),
            keras.callbacks.ModelCheckpoint("Model_VGG.h5",verbose=0,save_best_only=True)]

model.fit_generator( train_generator, epochs =10, validation_data=validation_generator, callbacks=my_calls)