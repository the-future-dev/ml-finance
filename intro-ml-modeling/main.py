from code.dataset import fetch_and_save_data, load_data, preprocess_data, exploratory_data_analysis, data_split, calculate_stats, normalize_data
from code.model import define_features_targets, build_model, load_model, train_model, evaluate_model, plot_loss_history
from code.strategy import trading_strategy, backtest_strategy
import os

if __name__ == "__main__":
    # Fetch and preprocess the data
    ticker = 'AAPL'
    start = '2003-09-30'
    end = '2023-09-30'
    filename = f"./datasets/{ticker}_{start}_{end}.csv"
    if not os.path.exists(filename):
        fetch_and_save_data(ticker, start, end, filename)
        print("Data downloaded and saved.")
    else:
        print("Data already existing in the machine")

    raw_data = load_data(filename, start, end)
    print("Data loaded")
    data = preprocess_data(raw_data)
    exploratory_data_analysis(data)
    
    # Define features and targets, and split the data
    X, y = define_features_targets(data)
    X_train, X_test, y_train, y_test = data_split(X, y, test_size=0.2)

    mean, std_dev = calculate_stats(X_train)

    X_train_norm = normalize_data(X_train, mean, std_dev)
    X_test_norm = normalize_data(X_test, mean, std_dev)

    # Hyperparameters
    learning_rate = 0.001
    patience = 6
    num_epochs = 1000000

    # Build and train the model
    model = build_model(X_train_norm.shape[1])  # passing the number of features as input_size
    
    # MODEL LOADING
    date_time = '20230930-195145'
    model = load_model(model, date_time)

    # MODEL TRAINING
    # model, loss_history = train_model(model, X_train_norm, y_train, learning_rate, num_epochs=num_epochs, patience=patience)

    # # Plotting loss history
    # plot_loss_history(loss_history)
    
    # Evaluate the model
    # evaluate_model(model, X_test_norm, y_test)

    history = trading_strategy(model, X)
    print(history.head())

    backtest_strategy(history)
