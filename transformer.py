import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import pickle
import time
import tarfile
from functools import reduce
from tensorflow.keras.layers import (Input, MultiHeadAttention, Dense, LayerNormalization, 
                                     Add, Dropout, PReLU, GlobalAveragePooling1D, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LambdaCallback, Callback
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam


def extract_tar_gz(file_path, destination):
    try:
        with tarfile.open(file_path, 'r:gz') as file:
            file.extractall(path=destination)
    except Exception as e:
        print(f"Error while extracting {file_path}: {e}")

if not os.path.exists("./features_of_interest") and os.path.exists("./features_of_interest.tar.gz"):
    print("Extracting features_of_interest.tar.gz...")
    extract_tar_gz("./features_of_interest.tar.gz", ".")

print("Reading CSV files...")
csv_files = glob.glob("./features_of_interest/*.csv")
print(f"Found {len(csv_files)} CSV files.")

sample_df = pd.read_csv(csv_files[0])
list_of_all_columns = sample_df.columns.tolist()

# Constants
BATCH_SIZE = 5
EPOCHS = 100
WEIGHTS_PATH = 'transformer_weights.h5'
TRAIN_DATASET_PATH = f"serialized_data/train_dataset_batch_{BATCH_SIZE}"
TEST_DATASET_PATH = f"serialized_data/test_dataset_batch_{BATCH_SIZE}"
os.makedirs(TRAIN_DATASET_PATH, exist_ok=True)
os.makedirs(TEST_DATASET_PATH, exist_ok=True)
FEATURE_COLS = [col_name for col_name in list_of_all_columns if col_name not in ["frameTime", "arousal", "valence"]]
TARGET_COLS = ["arousal", "valence"]

def load_datasets(file_paths):
    dfs = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(dfs, axis=0)
    return combined_df.copy()

def gather_statistics(df):
    mean = df[FEATURE_COLS].mean()
    std = df[FEATURE_COLS].std()
    return mean, std

def load_and_preprocess_dataset(df, mean, std):
    for col in FEATURE_COLS:
        df.loc[:, col] = (df[col] - mean[col]) / std[col]
    return df

def dataframe_to_dataset(df):
    feature_dict = dict(zip(FEATURE_COLS, df[FEATURE_COLS].values.T))
    features = tf.stack(list(feature_dict.values()), axis=-1)
    targets = tf.stack([df['arousal'].values, df['valence'].values], axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    return dataset

pickle_filename = 'all_datasets.pkl'
combined_df = load_datasets(csv_files)

dataset_size = len(combined_df)
BUFFER_SIZE = dataset_size
split_fraction = 0.8
train_size = int(split_fraction * dataset_size)

# Split the datasets into training and test first
train_df = combined_df.iloc[:train_size].copy()
test_df = combined_df.iloc[train_size:].copy()

# Gather statistics only on the training set
print("Gathering statistics...")
mean, std = gather_statistics(train_df)
print("Statistics gathered.")

if os.path.exists(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        all_dataframes = pickle.load(f)
    print("Loaded all_dataframes from pickle file.")
else:
    all_dataframes = [load_and_preprocess_dataset(df_chunk, mean, std) for df_chunk in [train_df, test_df]]
    with open(pickle_filename, 'wb') as f:
        pickle.dump(all_dataframes, f)
    print("All dataframes loaded, preprocessed, and saved to pickle file.")

all_dataframes = [df.astype('float64') for df in all_dataframes]
all_datasets = [dataframe_to_dataset(df) for df in all_dataframes]

def safe_concatenate(ds_list):
    combined = ds_list[0]
    for idx, ds in enumerate(ds_list[1:], start=1):
        try:
            combined = combined.concatenate(ds)
        except Exception as e:
            print(f"Error encountered while concatenating dataset at index {idx}.")
            for sample in ds.take(1):
                print(sample)
            raise e
    return combined

train_dataset, test_dataset = all_datasets

print("Checking for existing train and test datasets...")
if os.path.exists(os.path.join(TRAIN_DATASET_PATH, "dataset_spec.pb")) and os.path.exists(os.path.join(TEST_DATASET_PATH, "dataset_spec.pb")):
    print("Loading existing train and test datasets...")
    train_dataset = tf.data.Dataset.load(TRAIN_DATASET_PATH)
    test_dataset = tf.data.Dataset.load(TEST_DATASET_PATH)
    print("Existing datasets loaded.")
else:
    print("Splitting and shuffling datasets...")
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    print("Datasets split and shuffled.")
    print("Saving train and test datasets...")
    start_time = time.time()
    train_dataset.save(TRAIN_DATASET_PATH)
    test_dataset.save(TEST_DATASET_PATH)
    end_time = time.time()
    print(f"Datasets saved in {end_time - start_time} seconds.")


def get_positional_encoding(seq_len, d_model):
    angles = np.arange(seq_len)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    pos_encoding = np.zeros(angles.shape)
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_block(input_layer, d_model, num_heads, dff, rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(input_layer, input_layer)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(Add()([input_layer, attn_output]))
    ff_output = Dense(dff)(out1)
    ff_output = PReLU()(ff_output)
    ff_output = Dense(d_model)(ff_output)
    ff_output = Dropout(rate)(ff_output)
    out2 = LayerNormalization(epsilon=1e-6)(Add()([out1, ff_output]))
    return out2

def build_transformer_model(input_shape, num_heads, d_model, dff, rate=0.1, num_blocks=6, reg_strength=0.001):
    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs)
    pos_enc = get_positional_encoding(input_shape[0], d_model)
    x = Add()([x, pos_enc])
    for _ in range(num_blocks):
        x = transformer_block(x, d_model, num_heads, dff, rate)
    transformer_out = GlobalAveragePooling1D()(x)
    x = Dense(512)(transformer_out)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = PReLU()(x)
    x = BatchNormalization()(x)
    outputs = Dense(2, activation='tanh')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

class PrintLoss(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        print(f"\nEnd of Epoch {epoch + 1} — val_loss: {val_loss:.10f}\n")

d_model = 128
dff = 512
num_heads = 4

sample_features, _ = next(iter(train_dataset))
input_shape = sample_features.shape[1:]

checkpoint_dir = "model_checkpoints"

weights_file_path = os.path.join(checkpoint_dir, WEIGHTS_PATH)
print("Building model architecture...")
model = build_transformer_model(input_shape, num_heads, d_model, dff)

if os.path.exists(weights_file_path):
    print("Loading weights from saved checkpoint...")
    try:
        model.load_weights(weights_file_path)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
else:
    print("No weights found. Initializing model with random weights.")
    os.makedirs(checkpoint_dir, exist_ok=True)

opt = Adam(learning_rate=0.00001)
model.compile(optimizer=opt, loss='mse')
model.summary()

def save_best_val_loss(val_loss, filename="best_val_loss.txt"):
    with open(filename, 'w') as f:
        f.write(str(val_loss))

def load_best_val_loss(filename="best_val_loss.txt"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return float(f.read().strip())
    return float('inf')  # if file doesn't exist, set to infinity

best_val_loss_file = "best_val_loss.txt"

original_best_val_loss = load_best_val_loss(best_val_loss_file)

best_val_loss = original_best_val_loss

def custom_checkpoint(epoch, logs):
    global best_val_loss
    
    current_val_loss = logs.get('val_loss')

    # Check if current validation loss is less than the previous best
    if current_val_loss < best_val_loss:
        # Save the model weights
        model.save_weights(weights_file_path)

        # Update the in-memory best validation loss
        best_val_loss = current_val_loss
        print(f"\n***NEW BEST MODEL*** Epoch {epoch + 1} model weights saved with validation loss: {current_val_loss:.10f}\n\n")


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-11, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=EPOCHS, verbose=1, restore_best_weights=True)
custom_checkpoint_callback = LambdaCallback(on_epoch_end=custom_checkpoint)
callbacks_list = [PrintLoss(), custom_checkpoint_callback, early_stopping, reduce_lr]

try:
    # Training the model
    print(f"Current best loss: {best_val_loss}")
    print(f"Starting model training for {EPOCHS} epochs...")
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=EPOCHS, callbacks=callbacks_list, verbose=1)
    
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"Final Training Loss: {final_train_loss}")
    print(f"Final Validation Loss: {final_val_loss}")

except (Exception, KeyboardInterrupt) as e:
    print(f"Training was interrupted with error: {e}")

finally:
    # Only save if the in-memory best loss is better than the originally-loaded one
    if best_val_loss < original_best_val_loss:
        save_best_val_loss(best_val_loss, best_val_loss_file)
        print("\nUpdated best validation loss saved.\n")
    else:
        print("\nBest validation loss not improved from the original.\n")