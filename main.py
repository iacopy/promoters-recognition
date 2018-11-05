import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

import loader

NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

(x_train, y_train),(x_test, y_test) = loader.load_data(sys.argv[1])

model = Sequential()
model.add(Dense(512, input_dim=500))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=TRAIN_BATCH_SIZE)

score = model.evaluate(x_test, y_test, batch_size=TEST_BATCH_SIZE)
model.save('model.h5')
print("Test -  loss: {}, accuracy: {}".format(score[0], score[1]))
