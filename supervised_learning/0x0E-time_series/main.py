#!/usr/bin/env python3
"""
Contains the WindowGenerator Class
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


preprocessing = __import__('preprocess_data').preprocessing
WindowGenerator = __import__('forecast_btc').WindowGenerator
Baseline = __import__('forecast_btc').Baseline
build_model = __import__('forecast_btc').build_model
compile_and_fit = __import__('forecast_btc').compile_and_fit


if __name__ == '__main__':

    # PRE-PROCESSING
    train_df, val_df, test_df = preprocessing

    # Wide window
    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        label_columns=['Close'])
    column_indices = wide_window.column_indices
    print(wide_window)

    # Performance log
    val_performance = {}
    performance = {}

    # BASELINE
    baseline = Baseline(label_index=column_indices['Close'])

    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])

    # Evaluate  Baseline with wide_window
    val_performance['Baseline_ww'] = baseline.evaluate(wide_window.val)
    performance['Baseline_ww'] = baseline.evaluate(wide_window.test, verbose=0)

    print('Input shape:', wide_window.example[0].shape)
    print('Output shape:', baseline(wide_window.example[0]).shape)

    wide_window.plot(baseline)

    # LSTM
    lstm_model = build_model

    # Train LSTM with wide window
    epochs = 500
    history = compile_and_fit(lstm_model, wide_window)

    # Evaluate LSTM with wide_window
    val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
    performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)

    wide_window.plot(lstm_model)

    # PERFORMANCE COMPARISON
    x = np.arange(len(performance))
    width = 0.3
    metric_name = 'mean_absolute_error'
    metric_index = lstm_model.metrics_names.index('mean_absolute_error')
    val_mae = [v[metric_index] for v in val_performance.values()]
    test_mae = [v[metric_index] for v in performance.values()]

    plt.ylabel('mean_absolute_error [T (degC), normalized]')
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(),
               rotation=45)
    _ = plt.legend()

    for name, value in performance.items():
        print(f'{name:12s}: {value[1]:0.4f}')
