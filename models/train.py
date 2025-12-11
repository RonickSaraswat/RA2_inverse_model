# models/train.py
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from bi_lstm_model import build_bi_lstm_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
os.makedirs(MODELS_OUT, exist_ok=True)

# hyperparams
T = 10
batch_size = 32
epochs = 150
dropout_rate = 0.1
use_mc_dropout = False  # can be set true if dropout set for MC
smoothness_lambda = 0.0  # set >0 to enable smoothness 

# load
X = np.load(os.path.join(DATA_OUT, "features.npy"))  # (N, Ftotal)
y = np.load(os.path.join(DATA_OUT, "params.npy"))

# choose subset sizes for an experiment OR use full dataset
subset_size = None  # e.g., 1000, 5000 or None for full
if subset_size is not None:
    idx = np.random.choice(len(X), subset_size, replace=False)
    X = X[idx]
    y = y[idx]

# Prepare sequential shape for LSTM: split feature vector into T timesteps with equal F
F_total = X.shape[1]
if F_total % T != 0:
    # truncate to make divisible
    F = F_total // T
    X = X[:, :T*F]
else:
    F = F_total // T
X_seq = X.reshape(X.shape[0], T, F)

# standardize per feature (across samples and time collapsed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_seq.reshape(-1, F)).reshape(X_seq.shape)

# split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# build model
model = build_bi_lstm_model(T, F, y.shape[1], dropout_rate=dropout_rate, use_mc_dropout=use_mc_dropout)

# optional custom loss with smoothness penalty across parameters
if smoothness_lambda > 0:
    import tensorflow.keras.backend as K
    def custom_loss(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred))
        # smoothness: encourage small differences between adjacent predicted params
        smooth = K.mean(K.square(y_pred[:, 1:] - y_pred[:, :-1]))
        return mse + smoothness_lambda * smooth
    loss_fn = custom_loss
else:
    loss_fn = 'mse'

model.compile(optimizer=Adam(), loss=loss_fn, metrics=['mse'])

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[es])

# save model (native .keras)
model_path = os.path.join(MODELS_OUT, "bi_lstm_inverse_model.keras")
model.save(model_path)
print("Model saved to:", model_path)

# save scaler
with open(os.path.join(MODELS_OUT, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# save history
with open(os.path.join(MODELS_OUT, "training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)
print("Saved training history and scaler.")
