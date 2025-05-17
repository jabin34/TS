import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

input_file = "/home/israt/OMNETPP/ts/simu5G/src/data/tower_Load_test.txt"
output_file_4g = "/home/israt/OMNETPP/ts/simu5G/src/data/outputLSTM_4G.txt"
output_file_5g = "/home/israt/OMNETPP/ts/simu5G/src/data/outputLSTM_5G.txt"
n_steps = 10

try:
    df = pd.read_csv(input_file, header=None, names=['Time', 'TowerId', 'Type', 'AvgLoad'])
except FileNotFoundError:
    print(f"Error: {input_file} not found")
    exit(1)

df_4g = df[df['Type'] == 0]['AvgLoad'].values
df_5g = df[df['Type'] == 1]['AvgLoad'].values

scaler_4g = MinMaxScaler(feature_range=(0, 1))
scaler_5g = MinMaxScaler(feature_range=(0, 1))
df_4g_scaled = scaler_4g.fit_transform(df_4g.reshape(-1, 1)).flatten()
df_5g_scaled = scaler_5g.fit_transform(df_5g.reshape(-1, 1)).flatten()

if len(df_4g) < n_steps + 1 or len(df_5g) < n_steps + 1:
    print(f"Insufficient data - 4G: {len(df_4g)}, 5G: {len(df_5g)}")
    prediction_4g = prediction_5g = 0.1
else:
    def split_sequence(sequence, n_steps):
        X, y = [], []
        for i in range(len(sequence) - n_steps):
            X.append(sequence[i:i + n_steps])
            y.append(sequence[i + n_steps])
        return np.array(X), np.array(y)

    X_4g, y_4g = split_sequence(df_4g_scaled, n_steps)
    X_5g, y_5g = split_sequence(df_5g_scaled, n_steps)
    X_4g = X_4g.reshape((X_4g.shape[0], X_4g.shape[1], 1))
    X_5g = X_5g.reshape((X_5g.shape[0], X_5g.shape[1], 1))

    def train_lstm(X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = Sequential([
            LSTM(10, activation='tanh', input_shape=(n_steps, 1), return_sequences=True),
            Dropout(0.1),
            LSTM(10, activation='tanh'),
            Dense(5, activation='tanh'),
            Dense(1, activation='sigmoid')  # Cap output to [0, 1]
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32, verbose=0)
        return model

    model_4g = train_lstm(X_4g, y_4g)
    model_5g = train_lstm(X_5g, y_5g)

    test_4g = df_4g_scaled[-n_steps:].reshape((1, n_steps, 1))
    test_5g = df_5g_scaled[-n_steps:].reshape((1, n_steps, 1))
    prediction_4g_scaled = model_4g.predict(test_4g, verbose=0)[0][0]
    prediction_5g_scaled = model_5g.predict(test_5g, verbose=0)[0][0]

    prediction_4g = scaler_4g.inverse_transform([[prediction_4g_scaled]])[0][0]
    prediction_5g = scaler_5g.inverse_transform([[prediction_5g_scaled]])[0][0]

    prediction_4g = min(max(prediction_4g, 0.0), 1.0)
    prediction_5g = min(max(prediction_5g, 0.0), 1.0)

with open(output_file_4g, 'w') as f:
    f.write(f"{prediction_4g:.6f}")
with open(output_file_5g, 'w') as f:
    f.write(f"{prediction_5g:.6f}")

print(f"Predicted 4G Load: {prediction_4g:.6f}")
print(f"Predicted 5G Load: {prediction_5g:.6f}")
#
#
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.preprocessing import MinMaxScaler
# import os
# import time
#
# input_file = "/home/israt/OMNETPP/ts/simu5G/src/data/towerLoadData.txt"
# output_file_4g = "/home/israt/OMNETPP/ts/simu5G/src/data/outputLSTM_4G.txt"
# output_file_5g = "/home/israt/OMNETPP/ts/simu5G/src/data/outputLSTM_5G.txt"
# model_file_4g = "/home/israt/OMNETPP/ts/simu5G/src/data/lstm_model_4g.h5"
# model_file_5g = "/home/israt/OMNETPP/ts/simu5G/src/data/lstm_model_5g.h5"
# n_steps = 10
#
# def load_and_prepare_data():
#     try:
#         df = pd.read_csv(input_file, header=None, names=['Time', 'TowerId', 'Type', 'AvgLoad'])
#     except FileNotFoundError:
#         return None, None
#
#     df_4g = df[df['Type'] == 0]['AvgLoad'].values
#     df_5g = df[df['Type'] == 1]['AvgLoad'].values
#
#     scaler_4g = MinMaxScaler(feature_range=(0, 1))
#     scaler_5g = MinMaxScaler(feature_range=(0, 1))
#     df_4g_scaled = scaler_4g.fit_transform(df_4g.reshape(-1, 1)).flatten()
#     df_5g_scaled = scaler_5g.fit_transform(df_5g.reshape(-1, 1)).flatten()
#
#     return (df_4g_scaled, scaler_4g), (df_5g_scaled, scaler_5g)
#
# def split_sequence(sequence, n_steps):
#     X, y = [], []
#     for i in range(len(sequence) - n_steps):
#         X.append(sequence[i:i + n_steps])
#         y.append(sequence[i + n_steps])
#     return np.array(X), np.array(y)
#
# def build_lstm():
#     model = Sequential([
#         LSTM(20, activation='tanh', input_shape=(n_steps, 1), return_sequences=True),
#         Dropout(0.1),
#         LSTM(10, activation='tanh'),
#         Dense(5, activation='tanh'),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     return model
#
# def predict_load(data_scaled, scaler, model_file):
#     if len(data_scaled) < n_steps + 1:
#         return 0.1
#
#     if os.path.exists(model_file):
#         model = load_model(model_file)
#     else:
#         X, y = split_sequence(data_scaled, n_steps)
#         if len(X) < 2:
#             return 0.1
#         X = X.reshape((X.shape[0], X.shape[1], 1))
#         model = build_lstm()
#         model.fit(X, y, epochs=20, batch_size=16, verbose=0)
#         model.save(model_file)
#
#     test_data = data_scaled[-n_steps:].reshape((1, n_steps, 1))
#     prediction_scaled = model.predict(test_data, verbose=0)[0][0]
#     prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]
#     return min(max(prediction, 0.0), 1.0)
#
# def main():
#     start_time = time.time()
#     (df_4g_scaled, scaler_4g), (df_5g_scaled, scaler_5g) = load_and_prepare_data()
#
#     if df_4g_scaled is None or df_5g_scaled is None:
#         prediction_4g = prediction_5g = 0.1
#     else:
#         prediction_4g = predict_load(df_4g_scaled, scaler_4g, model_file_4g)
#         prediction_5g = predict_load(df_5g_scaled, scaler_5g, model_file_5g)
#
#     with open(output_file_4g, 'w') as f:
#         f.write(f"{prediction_4g:.6f}")
#     with open(output_file_5g, 'w') as f:
#         f.write(f"{prediction_5g:.6f}")
#
#     print(f"Predicted 4G Load: {prediction_4g:.6f}, 5G Load: {prediction_5g:.6f}")
#     print(f"Execution time: {time.time() - start_time:.2f} seconds")
#
# if __name__ == "__main__":
#     main()