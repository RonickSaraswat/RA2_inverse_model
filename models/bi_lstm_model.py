# models/bi_lstm_model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout
import tensorflow as tf

def build_bi_lstm_model(T, F, P, dropout_rate=0.1, use_mc_dropout=False):
    """
    If use_mc_dropout True, model includes Dropout layers that can be left enabled during inference (training=True).
    """
    inp = Input(shape=(T, F))
    x = Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid',
                           dropout=dropout_rate, return_sequences=False))(inp)
    if use_mc_dropout:
        # For adding an extra dropout layer
        x = Dropout(dropout_rate)(x, training=True)  # force MC behaviour (training flag used at prediction)
    out = Dense(P, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    return model
