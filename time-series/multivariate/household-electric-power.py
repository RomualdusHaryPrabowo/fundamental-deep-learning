import pandas as pd
import tensorflow as tf

# 1. Load Data
df = pd.read_csv(
    "https://drive.google.com/uc?id=1AZRfFoyekqSYpri5183RmJjciRGz_ood",
    sep=",",
    index_col="datetime",
    header=0,
)

# 2. Normalisasi (Min-Max Scaling agar data berada di rentang 0-1)
def normalize_series(data, min_val, max_val):
    data = (data - min_val) / (max_val - min_val)
    return data

data = df.values
data = normalize_series(data, data.min(axis=0), data.max(axis=0))

N_FEATURES = len(df.columns)

# 3. Split Data (50% train, 50% valid)
SPLIT_TIME = int(len(data) * 0.5)
x_train = data[:SPLIT_TIME]
x_valid = data[SPLIT_TIME:]

# 4. Fungsi Windowing (Membentuk input dan target)
def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    # Membuat window geser: ukuran total (past + future)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    # Memisahkan n_past sebagai input X, dan n_future sebagai target Y
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

BATCH_SIZE = 32
N_PAST = 24
N_FUTURE = 24
SHIFT = 1

train_set = windowed_dataset(x_train, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT)
valid_set = windowed_dataset(x_valid, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT)

# 5. Arsitektur Model (Dense/MLP)
model = tf.keras.models.Sequential([
    # Input shape: (24, N_FEATURES), Dense akan mem-flatten data ini
    tf.keras.layers.Flatten(input_shape=(N_PAST, N_FEATURES)), 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    # Output layer harus menghasilkan n_future * N_FEATURES
    tf.keras.layers.Dense(N_FUTURE * N_FEATURES),
    tf.keras.layers.Reshape([N_FUTURE, N_FEATURES]) # Mengembalikan bentuk ke (24, N_FEATURES)
])

# 6. Callback untuk efisiensi training
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('mae') < 0.055 and logs.get('val_mae') < 0.055):
            print("\nMAE target tercapai, menghentikan training!")
            self.model.stop_training = True

callbacks = myCallback()

# 7. Kompilasi dan Training
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mae', optimizer=optimizer, metrics=["mae"])

model.fit(train_set, validation_data=valid_set, epochs=100, callbacks=callbacks)

train_pred = model.predict(train_set)
print (train_pred[0][0])