from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
import tensorflow_addons as tfa


# Builds the RNN model.
def build_model(input_shape, layers, show_summary=True):
    model = Sequential()
    model.add(Input(shape=input_shape))
    for layer in layers:
        model.add(layer)

    model.compile(optimizer=tfa.optimizers.Yogi(learning_rate=0.001), loss='huber')

    if show_summary:
        model.summary()

    return model


# Fits the data into RNN.
def train(model, name, x_train, y_train, epochs=200, batch_size=32):
    checkpoint = ModelCheckpoint(filepath='weights/' + name + '_callback.h5',
                                 monitor='loss',
                                 mode='min',
                                 save_best_only=True,
                                 verbose=1)
    early_stopping = EarlyStopping(monitor='loss',
                                   min_delta=0,
                                   patience=25,
                                   verbose=1,
                                   restore_best_weights=True)

    model_callbacks = [checkpoint, early_stopping]

    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=model_callbacks, verbose=1)
    return model, history

