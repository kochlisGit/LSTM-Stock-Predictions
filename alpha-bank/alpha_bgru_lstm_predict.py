from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM, GRU
import rnn_model
import stocks
import tensorflow_addons as tfa

# Reading & Preprocessing data.
_, test_data = stocks.read_data(preprocessing=True, scaling_method='MinMax')

x_test, y_test = stocks.construct_time_frames(test_data, frame_size=16)

# Defining layers of LSTM.
input_shape=x_test.shape[1:]
layers = [
    Bidirectional(GRU(units=50, return_sequences=True)),
    tfa.layers.GroupNormalization(),
    Dropout(0.2),
    Bidirectional(GRU(units=50, return_sequences=True)),
    tfa.layers.GroupNormalization(),
    Dropout(0.2),
    Bidirectional(LSTM(units=50)),
    tfa.layers.GroupNormalization(),
    Dropout(0.2),
    Dense(units=1, activation='sigmoid')
]

# Making predictions.
model = rnn_model.build_model(input_shape, layers)
model.load_weights('weights/bgru_lstm_callback.h5')

y_predict = model.predict(x_test)

# Visualising the prediction.
stocks.plot_prediction(y_test, y_predict, 'Real ALPHA BANK Stock Price', 'Predicted ALPHA BANK Stock Price')

for i, y_pred in enumerate(y_predict):
    print('Day', i, ':', y_pred[0])
