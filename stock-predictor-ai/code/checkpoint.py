# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Conv1DTranspose, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras.regularizers import l2

from shared import mean_abs_error

def read_dataset(filename, start, end):
    data = pd.read_csv(filename, index_col=0, parse_dates=True)
    return data.loc[start:end]

def plot_results(price_actual_array, price_predicted_array, target_column_name, title="Actual vs Predicted"):
    plt.figure(figsize=(15, 6))
    plt.plot(price_actual_array, label="Actual Price", color='blue')
    plt.plot(price_predicted_array, label="Predicted Price", color='red', linestyle='dashed')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel(target_column_name)
    plt.legend()

    plt.show()


def evaluate_price_predictions(price_actual_array, price_predicted_array):
    """ 
    Evaluation function: given two arrays, one for the price prediction and one for the actual price,
    evaluates the performance using various metrics.
    """
    # should this be included?
    price_actual_array = np.array(price_actual_array)
    price_predicted_array = np.array(price_predicted_array)
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(price_predicted_array - price_actual_array))
    
    # Mean Squared Error (MSE)
    mse = np.mean((price_predicted_array - price_actual_array)**2)
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((price_actual_array - price_predicted_array) / price_actual_array)) * 100
    
    # R-squared (R²)
    # Total sum of squares (SST) and residual sum of squares (SSR)
    sst = np.sum((price_actual_array - np.mean(price_actual_array))**2)
    ssr = np.sum((price_actual_array - price_predicted_array)**2)
    r_squared = 1 - (ssr/sst)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared (R²): {r_squared:.2f}")



# %% [markdown]
# ### Data fetching

# %%
start = "2021-01-01"
end = "2023-10-01"
target_column_name = 'High'
path = '../models/lstm_model/predictor_adj_close.h5'

AZM = read_dataset('../data/AZM.MI_ta.csv', start, end)

data = AZM
data.dropna(inplace=True)

cols = [target_column_name] + [ col for col in data if col != target_column_name]
target_column = list(data.columns).index(target_column_name)
data = data[cols]

print(f"#Trading Days: {data.shape}")

# %% [markdown]
# ### Data refactoring

# %%
# Define feature array and target array to train the model.
data_array = np.array(data.values)
target_array = np.array(data[target_column_name].values).reshape(-1, 1)

# Normalize the data
scaler_data = MinMaxScaler()
scaler_data.fit(data_array)
data_array = scaler_data.transform(data_array)

scaler_target = MinMaxScaler()
scaler_target.fit(target_array)
target_array = scaler_target.transform(target_array)

# Split the data
train_size = int(len(data_array) * 0.90)

def create_sequences(data, target, seq_length):
    sequence_data = []
    sequence_target = []
    for i in range(seq_length, len(data)):
        sequence_data.append(data[i-seq_length:i])
        sequence_target.append(target[i])
    return np.array(sequence_data), np.array(sequence_target)

SEQUENCE_LENGTH = 17
data_sequences, target_sequences = create_sequences(data_array, target_array, SEQUENCE_LENGTH)

train_data, test_data = data_sequences[:train_size], data_sequences[train_size:]
train_target, test_target = target_sequences[:train_size], target_sequences[train_size:]

print(f'train_data: {train_data.shape} triat_target: {train_target.shape}')
print(f'test_data: {test_data.shape} test_target: {test_target.shape}')

# %% [markdown]
# ## Model definition
# We will use three different types of Deep Neural Networks:
#  - LSTM
#  - CNN
#  - Dense
# 
# At first we will train independently the CNN through a VAE and we will use the encoder for feature extrapolation.
# Then we'll combine the three branches and train the model.
# 
# ### VAE (Variational Auto Encoder)
# As a feature extractor for our main neural network.

# %%
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_encoder(sequence_length, n_features, latent_dim):
    inputs = Input(shape=(sequence_length, n_features))
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Flatten()(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dropout(0.7)(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    return encoder

def create_decoder(sequence_length, n_features, latent_dim):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(sequence_length * n_features, activation='relu')(latent_inputs)
    x = Reshape((sequence_length, n_features))(x)
    # x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    outputs = Conv1D(filters=n_features, kernel_size=3, activation='sigmoid', padding='same')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()
        self.beta = beta

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampling((z_mean, z_log_var))
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction), axis=1
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = self.beta * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))  # Weighted KL loss
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    
    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var = self.encoder(data)
        z = self.sampling((z_mean, z_log_var))
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.mean_squared_error(data, reconstruction), axis=1
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

def build_autoencoder(encoder, decoder, learning_rate, beta):
    vae = VAE(encoder, decoder, beta=beta)
    vae.compile(optimizer=Adam(learning_rate=learning_rate))
    return vae


# %% [markdown]
# ### Build and Train the model

# %%
latent_dim = 69
epochs = 300
batch_size = 128
patience = 20
encoder = create_encoder(sequence_length=train_data.shape[1], n_features=train_data.shape[2], latent_dim=latent_dim)
decoder = create_decoder(sequence_length=train_data.shape[1], n_features=train_data.shape[2], latent_dim=latent_dim)

vae = build_autoencoder(encoder, decoder, learning_rate=0.0003, beta=0.3)

print("\n")
test_loss = vae.evaluate(test_data, test_data)
print("\n")

early_stopping = keras.callbacks.EarlyStopping(
    monitor='kl_loss',
    patience=patience,
    verbose=2,
    mode='min',
    restore_best_weights=False,
)

vae.fit(train_data, train_data, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[early_stopping])

# %%
test_loss = vae.evaluate(test_data, test_data)

# %% [markdown]
# Loss of the test_data on the CNN before training:<br>
# > loss: 9.6933 - reconstruction_loss: 1.1233 - kl_loss: 8.5700
# 
# Loss of the test_data on the CNN post training: <br>
# > loss: 0.4672 - reconstruction_loss: 0.4509 - kl_loss: 0.0163

# %% [markdown]
# ### Main Model Function Definition
# Three branches with independent input all converge to a dense layer.

# %%
def build_parallel_model(input_shape, encoder, l2_value=0.01):
    # LSTM Branch
    lstm_input = keras.layers.Input(shape=input_shape)
    lstm_branch = keras.layers.LSTM(128, kernel_regularizer=l2(l2_value), recurrent_regularizer=l2(l2_value), bias_regularizer=l2(l2_value))(lstm_input)

    # Dense Branch
    dense_input = keras.layers.Input(shape=input_shape)
    flattened = keras.layers.Flatten()(dense_input)
    dense_branch = keras.layers.Dense(90, activation='relu', kernel_regularizer=l2(l2_value))(flattened)
    dense_branch = keras.layers.Dense(10)(dense_branch)

    # VAE Branch
    vae_input = keras.layers.Input(shape=input_shape)
    z_mean, z_log_var = encoder(vae_input) # Use only z_mean for subsequent layers
    encoded_output = keras.layers.Flatten()(z_mean)

    # Combining the branches
    combined = keras.layers.concatenate([lstm_branch, dense_branch, encoded_output])

    # Additional layers after combining
    combined_dense = keras.layers.Dense(units=64, activation='relu',kernel_regularizer=l2(l2_value))(combined)
    output = keras.layers.Dense(1)(combined_dense)

    model = keras.models.Model(inputs=[lstm_input, dense_input, vae_input], outputs=output)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, train_data, train_target, epochs=30, batch_size=256, patience=20):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=2,
        mode='min',
        restore_best_weights=True,
    )

    # live_plot = LivePlotCallback()

    model.fit(
        [train_data, train_data, train_data],  # Assuming both branches use the same training data
        train_target,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        validation_split=0.2,
        callbacks=[early_stopping]
    )

# %%
batch_size = 1024
epochs = 3000
patience = 50
l2_value = 0.06

## GENERATE MODEL ##
model = build_parallel_model(input_shape=(train_data.shape[1], train_data.shape[2]), encoder=encoder, l2_value=l2_value)
# model = load_model("../models/3branches.h5")
train_model(model, train_data, train_target, epochs=epochs, batch_size=batch_size, patience=patience)


# %% [markdown]
# ### Evaluate the model

# %%
data_to_predict = (test_data, test_data, test_data)
actual_prediction = test_target

model.evaluate(data_to_predict, actual_prediction)

# TESTing: DENORMALIZE TARGET AND PREDICTIONS ##
price_predicted_array = scaler_target.inverse_transform(model.predict(data_to_predict)) #[1:]
price_actual_array = scaler_target.inverse_transform(actual_prediction).flatten() #[:-1]

## Evaluation
evaluate_price_predictions(price_predicted_array.flatten(), price_actual_array)

## PLOTting #
plot_results(price_actual_array, price_predicted_array, target_column_name)

# %% [markdown]
# ### Model training:
# 1st with 2010-01-01 - 2023-01-01 data
# -> then save
# 
# 2nd use 2023-01-01  - 2023-11-18 data
# -> override

# %%
model.save("../models/3branches/low.h5")

# %% [markdown]
# 


