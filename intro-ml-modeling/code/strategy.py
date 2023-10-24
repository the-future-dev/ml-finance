import torch
import matplotlib.pyplot as plt
import pandas as pd

def trading_strategy(model, X, initial_balance=100000):
    """
    Execute a basic trading strategy based on model predictions.

    Parameters:
    - model (nn.Module): A trained PyTorch model.
    - X (pd.DataFrame): Features for the model.
    - initial_balance (float): The initial amount of money to start with.

    Returns:
    - history (pd.DataFrame): A DataFrame containing the balance and position at each timestep.
    """
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Ensure model is in evaluation mode
    model.eval()

    # Initialize variables
    balance = initial_balance
    position = 0
    history = []
    
    # Convert X to tensor
    X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)

    # Execute the trading strategy
    with torch.no_grad():
        for t in range(len(X)-1):  # We omit the last timestep since we don't have the next day's price
            # Get model prediction
            prediction_tomorrow = model(X_tensor[t].unsqueeze(0)).item()
            prediction_today = model(X_tensor[t-1].unsqueeze(0)).item() if t > 0 else None

            # BUY signal
            if prediction_tomorrow > X['Close'].iloc[t] and balance > X['Close'].iloc[t]:
                position = balance // X['Close'].iloc[t]
                balance = balance - position * X['Close'].iloc[t]
            # SELL signal
            elif prediction_tomorrow < X['Close'].iloc[t] and position > 0:
                balance = balance + position * X['Close'].iloc[t]
                position = 0
            # HOLD or DO NOTHING

            # Log history
            history.append({
                'balance': balance,
                'position': position,
                'total': balance + position * X['Close'].iloc[t],
                'prediction': prediction_today,
                'close_price': X['Close'].iloc[t],
                'action': 'buy' if position > 0 else 'sell' if position == 0 and balance < initial_balance else 'hold'
            })
    
    return pd.DataFrame(history)

def backtest_strategy(history):
    """
    Visualize the performance of the trading strategy.

    Parameters:
    - history (pd.DataFrame): A DataFrame containing the balance and position at each timestep.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history['total'], label='Total Value')
    plt.plot(history['close_price'], label='Close Price')
    plt.title('Backtesting Result')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()