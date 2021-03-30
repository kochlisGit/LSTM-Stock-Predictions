from tensorflow.keras.layers import Dense, Dropout, GRU
import rnn_model
import stocks
import tensorflow_addons as tfa

# Reading & Preprocessing data.
train_data, test_data = stocks.read_data(preprocessing=True, scaling_method='MinMax')

# Creating time frames.
x_train, y_train = stocks.construct_time_frames(train_data, frame_size=64)
x_test, y_test = stocks.construct_time_frames(test_data, frame_size=64)

# Defining layers of LSTM.
input_shape=x_train.shape[1:]
layers = [
    GRU(units=50, return_sequences=True),
    tfa.layers.GroupNormalization(),
    Dropout(0.2),
    GRU(units=50, return_sequences=True),
    tfa.layers.GroupNormalization(),
    Dropout(0.2),
    GRU(units=50),
    tfa.layers.GroupNormalization(),
    Dropout(0.2),
    Dense(units=1, activation='sigmoid')
]
name = 'gru'
epochs = 200
batch_size = 32

# Training the model.
model = rnn_model.build_model(input_shape, layers)
model, history = rnn_model.train(model, name, x_train, y_train, epochs=epochs, batch_size=batch_size)

# Making predictions.
y_predict = model.predict(x_test)

# Visualising the prediction.
stocks.plot_prediction(y_test, y_predict, 'Real Tesla Stock Price', 'Predicted Tesla Stock Price')
