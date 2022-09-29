import os

import matplotlib
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('Qt5Agg')


def load_data(window_size):
    raw_data_df = pd.read_csv("./AAPL.csv", index_col="Date")

    scaler = StandardScaler()
    raw_data = scaler.fit_transform(raw_data_df)
    plot_data = {"mean": scaler.mean_[3], "var": scaler.var_[3], "date": raw_data_df.index}

    raw_X = raw_data[:, :4]
    raw_y = raw_data[:, 3]

    X, y = [], []
    for i in range(len(raw_X) - window_size):
        cur_prices = raw_X[i:i + window_size, :]
        target = raw_y[i + window_size]

        X.append(list(cur_prices))
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    total_len = len(X)
    train_len = int(total_len * 0.8)

    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    return X_train, X_test, y_train, y_test, plot_data


def build_rnn_model(window_size, num_features):
    model = Sequential()

    model.add(layers.SimpleRNN(256, input_shape=(window_size, num_features)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))

    return model


def build_lstm_model(window_size, num_features):
    model = Sequential()

    model.add(layers.LSTM(256, input_shape=(window_size, num_features)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))

    return model


def build_gru_model(window_size, num_features):
    model = Sequential()

    model.add(layers.GRU(256, input_shape=(window_size, num_features)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))

    return model


def run_model(model, X_train, X_test, y_train, y_test, epochs=10, name=None):
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse")

    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=128, shuffle=True, verbose=2)

    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print("[{}] 테스트 loss: {:.5f}".format(name, test_loss))
    print()

    return optimizer, hist


def plot_result(model, X_true, y_true, plot_data, name):
    y_pred = model.predict(X_true)

    y_true_orig = (y_true * np.sqrt(plot_data["var"])) + plot_data["mean"]
    y_pred_orig = (y_pred * np.sqrt(plot_data["var"])) + plot_data["mean"]

    test_date = plot_data["date"][-len(y_true):]

    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.plot(y_true_orig, color="b", label="True")
    ax.plot(y_pred_orig, color="r", label="Prediction")
    ax.set_xticks(list(range(len(test_date))))
    ax.set_xticklabels(test_date, rotation=45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.set_title("{} Result".format(name))
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("apple_stock_{}".format(name.lower()))
    plt.show()


def main():
    tf.set_random_seed(2022)

    window_size = 30
    X_train, X_test, y_train, y_test, plot_data = load_data(window_size)
    num_features = X_train[0].shape[1]

    rnn_model = build_rnn_model(window_size, num_features)
    lstm_model = build_lstm_model(window_size, num_features)
    gru_model = build_gru_model(window_size, num_features)

    run_model(rnn_model, X_train, X_test, y_train, y_test, name="RNN")
    run_model(lstm_model, X_train, X_test, y_train, y_test, name="LSTM")
    run_model(gru_model, X_train, X_test, y_train, y_test, name="GRU")

    plot_result(rnn_model, X_test, y_test, plot_data, name="RNN")
    plot_result(lstm_model, X_test, y_test, plot_data, name="LSTM")
    plot_result(gru_model, X_test, y_test, plot_data, name="GRU")


if __name__ == "__main__":
    main()
