# Define the autoencoder architecture
input_dim = train_data.shape[1]
encoding_dim = int(input_dim / 2)

autoencoder = Sequential([
    Input(shape=(input_dim,)),
    Dense(encoding_dim, activation='relu'),
    Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder model
autoencoder.fit(train_data, train_data, epochs=50, batch_size=256, validation_split=0.1, verbose=1)

# Calculate the reconstruction error for the entire dataset
reconstructed_entire_data = autoencoder.predict(scaled_entire_data)
mse = np.mean(np.power(scaled_entire_data - reconstructed_entire_data, 2), axis=1)

# Define the autoencoder architecture with more layers and neurons
input_dim = train_data.shape[1]
encoding_dim_1 = int(input_dim * 0.75)
encoding_dim_2 = int(input_dim * 0.5)
encoding_dim_3 = int(input_dim * 0.25)

autoencoder = Sequential([
    Input(shape=(input_dim,)),
    Dense(encoding_dim_1, activation='relu'),
    Dense(encoding_dim_2, activation='relu'),
    Dense(encoding_dim_3, activation='relu'),
    Dense(encoding_dim_2, activation='relu'),
    Dense(encoding_dim_1, activation='relu'),
    Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder model for more epochs
autoencoder.fit(train_data, train_data, epochs=100, batch_size=256, validation_split=0.1, verbose=1)

# Calculate the reconstruction error for the entire dataset
reconstructed_entire_data = autoencoder.predict(scaled_entire_data)
mse = np.mean(np.power(scaled_entire_data - reconstructed_entire_data, 2), axis=1)
