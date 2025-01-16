import matplotlib.pyplot as plt
import plotly.express as px

def plot_stock_predictions(df, predictions, title="Stock Price Predictions"):
    """
    Quick example using matplotlib to compare actual vs. predicted.
    Assumes df has a 'close' column, and predictions is aligned at the tail.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Actual')
    pred_index = range(len(df) - len(predictions), len(df))
    plt.plot(df.index[pred_index], predictions, label='Predicted', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.show()

def create_plotly_forecast_plot(df, predictions):
    """
    Example using Plotly to visualize forecast.
    """
    fig = px.line()
    fig.add_scatter(x=df.index, y=df['close'], mode='lines', name='Actual')
    pred_index = range(len(df) - len(predictions), len(df))
    fig.add_scatter(x=df.index[pred_index], y=predictions, mode='lines', name='Predicted')
    fig.show()
