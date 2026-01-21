# models/train.py
import os
import sys
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

DATA_OUT = os.path.join(BASE_DIR, "data_out")
MODELS_OUT = os.path.join(BASE_DIR, "models_out")
os.makedirs(MODELS_OUT, exist_ok=True)

from models.bi_lstm_model import build_bi_lstm_model
from models.param_transforms import theta_to_z

np.random.seed(42)
tf.random.set_seed(42)

batch_size = 32
epochs = 150
learning_rate = 1e-3
dropout_rate = 0.15  # slightly higher often helps
weight_decay = 1e-4
clipnorm = 1.0
augment_std = 0.02   # feature-space augmentation after scaling (small)

d_model = 128
num_layers = 4
num_heads = 4
ff_dim = 256

print("Loading data...")
X = np.load(os.path.join(DATA_OUT, "features.npy"), mmap_mode="r")      # (N, tokens, feat)
y_theta = np.load(os.path.join(DATA_OUT, "params.npy"), mmap_mode="r")  # (N, P)

meta = np.load(os.path.join(DATA_OUT, "tfr_meta.npz"))
param_meta = np.load(os.path.join(DATA_OUT, "param_meta.npz"))

param_names = [x.decode("utf-8") for x in param_meta["param_names"]]
low = param_meta["prior_low"].astype(np.float32)
high = param_meta["prior_high"].astype(np.float32)

N, n_tokens, feature_dim = X.shape
P = y_theta.shape[1]
n_time_patches = int(meta["n_time_patches"])
n_freq_patches = int(meta["n_freq_patches"])
if "n_tokens_erp" in meta.files:
    n_tokens_erp = int(meta["n_tokens_erp"])
else:
    # fallback for old meta files: ERP tokens should equal n_time_patches
    n_tokens_erp = int(meta["n_time_patches"])


print("X:", X.shape, "y:", y_theta.shape)
print("Params:", param_names)
print("Tokens:", n_tokens, "feature_dim:", feature_dim)
print("TF grid:", n_time_patches, "x", n_freq_patches, " ERP tokens:", n_tokens_erp)

y_z = theta_to_z(np.asarray(y_theta, dtype=np.float32), low, high)

train_idx, val_idx, y_train, y_val = train_test_split(
    np.arange(N), y_z, test_size=0.15, random_state=42
)

print("Fitting StandardScaler on training tokens...")
scaler = StandardScaler()
chunk = 256
for start in range(0, len(train_idx), chunk):
    sl = train_idx[start:start + chunk]
    flat = np.asarray(X[sl], dtype=np.float32).reshape(-1, feature_dim)
    scaler.partial_fit(flat)

def scale_batch(X_batch):
    flat = X_batch.reshape(-1, feature_dim)
    flat_s = scaler.transform(flat).astype(np.float32)
    return flat_s.reshape(X_batch.shape[0], n_tokens, feature_dim)

def make_dataset(indices, y, shuffle=False):
    indices = np.asarray(indices, dtype=np.int64)

    def gen():
        for i, idx in enumerate(indices):
            xb = np.asarray(X[idx:idx+1], dtype=np.float32)
            xb = scale_batch(xb)[0]
            yield xb, y[i].astype(np.float32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(n_tokens, feature_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(P,), dtype=tf.float32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    if shuffle and augment_std > 0:
        ds = ds.map(
            lambda x, y: (x + tf.random.normal(tf.shape(x), stddev=augment_std), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds

train_ds = make_dataset(train_idx, y_train, shuffle=True).repeat()
val_ds = make_dataset(val_idx, y_val, shuffle=False).repeat()

steps_per_epoch = int(np.ceil(len(train_idx) / batch_size))
validation_steps = int(np.ceil(len(val_idx) / batch_size))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)

def gaussian_nll_z(y_true, y_pred):
    mu = y_pred[:, :P]
    logvar = y_pred[:, P:]
    logvar = tf.clip_by_value(logvar, -10.0, 10.0)
    inv_var = tf.exp(-logvar)
    nll = 0.5 * (inv_var * tf.square(y_true - mu) + logvar)
    return tf.reduce_mean(tf.reduce_sum(nll, axis=1))

model = build_bi_lstm_model(
    n_tokens=n_tokens,
    feature_dim=feature_dim,
    n_params=P,
    n_time_patches=n_time_patches,
    n_freq_patches=n_freq_patches,
    n_tokens_erp=n_tokens_erp,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    ff_dim=ff_dim,
    dropout_rate=dropout_rate,
    return_attention=False,
)
model.summary()

# AdamW if available, else Adam
try:
    from tensorflow.keras.optimizers import AdamW
    opt = AdamW(learning_rate=learning_rate, weight_decay=weight_decay, clipnorm=clipnorm)
except Exception:
    opt = Adam(learning_rate=learning_rate, clipnorm=clipnorm)

model.compile(optimizer=opt, loss=gaussian_nll_z)

early_stop = EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

print("Training...")
hist = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[early_stop],
    verbose=1,
)

model_path = os.path.join(MODELS_OUT, "jr_paramtoken_inverse_model.keras")
model.save(model_path)
print("Saved model:", model_path)

with open(os.path.join(MODELS_OUT, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

with open(os.path.join(MODELS_OUT, "training_history.pkl"), "wb") as f:
    pickle.dump(hist.history, f)

np.savez(
    os.path.join(MODELS_OUT, "param_bounds.npz"),
    param_names=np.array(param_names, dtype="S"),
    prior_low=low,
    prior_high=high,
)
print("Saved scaler, history, and param bounds.")
