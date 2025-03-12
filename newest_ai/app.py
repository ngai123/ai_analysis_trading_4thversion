import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import yfinance as yf
import threading
import time
from datetime import datetime
import plotly
import plotly.graph_objs as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Import the trading system components
from ai_trading_analysis_4 import AITradingSystem, AdvancedTradingEnvironment, PrioritizedReplayBuffer

app = Flask(__name__)
app.secret_key = 'bitcoin_trading_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store state
training_tasks = {}
trained_models = {}
pretrained_model = None

def get_default_parameters():
    """Return default parameters for the trading system"""
    return {
        'ticker': 'BTC-USD',
        'start_date': (datetime.now().replace(year=datetime.now().year-4)).strftime('%Y-%m-%d'),
        'end_date': datetime.now().strftime('%Y-%m-%d'),
        'seq_length': 60,
        'initial_cash': 10000,
        'max_position': 0.5,
        'transaction_fee': 0.001,
        'use_continuous_action_space': False,
        'num_episodes': 10,  # Default to just 10 episodes for quick testing
        'model_path': r"C:\Users\User\newest_ai\saved_models\1000_bitcoin_model.keras",
        'scaler_path': r"C:\Users\User\newest_ai\saved_models\1000_scaler.json"
    }

@app.route('/')
def index():
    """Render the main page"""
    parameters = get_default_parameters()
    
    # Check if pre-trained model exists using absolute paths
    pretrained_model_path = r"C:\Users\User\newest_ai\saved_models\1000_bitcoin_model.keras"
    pretrained_scaler_path = r"C:\Users\User\newest_ai\saved_models\1000_scaler.json"
    has_pretrained_model = os.path.exists(pretrained_model_path) and os.path.exists(pretrained_scaler_path)
    
    return render_template('index.html', parameters=parameters, 
                           has_pretrained_model=has_pretrained_model)

@app.route('/api/available_tickers')
def available_tickers():
    """Return a list of available cryptocurrency tickers"""
    # Common crypto tickers
    crypto_tickers = [
        'BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD',
        'ADA-USD', 'DOT-USD', 'LINK-USD', 'XLM-USD', 'DOGE-USD'
    ]
    return jsonify(tickers=crypto_tickers)

@app.route('/api/check_model_paths')
def check_model_paths():
    """Check if the model file paths exist and are accessible"""
    model_path = r"C:\Users\User\newest_ai\saved_models\1000_bitcoin_model.keras"
    scaler_path = r"C:\Users\User\newest_ai\saved_models\1000_scaler.json"
    
    model_exists = os.path.exists(model_path)
    scaler_exists = os.path.exists(scaler_path)
    
    return jsonify({
        'status': 'success',
        'model_path': model_path,
        'model_exists': model_exists,
        'scaler_path': scaler_path,
        'scaler_exists': scaler_exists,
        'both_exist': model_exists and scaler_exists
    })

@app.route('/api/fetch_price_data', methods=['POST'])
def fetch_price_data():
    """Fetch historical price data for a given ticker and date range"""
    data = request.get_json()
    ticker = data.get('ticker', 'BTC-USD')
    start_date = data.get('start_date', '2020-01-01')
    end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    try:
        # Fetch data from Yahoo Finance
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Handle multi-index columns that yfinance returns
        if isinstance(df.columns, pd.MultiIndex):
            # Extract the first level of the multi-index
            df.columns = df.columns.get_level_values(0)
        
        # Format for plotting
        df = df.reset_index()
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Create a simple Plotly figure
        fig = px.line(df, x='Date', y='Close', title=f'{ticker} Price History')
        
        # Convert to JSON
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        return jsonify({
            'status': 'success',
            'data': df.to_dict('records'),
            'plot': plot_json,
            'ticker': ticker,
            'start_date': start_date,
            'end_date': end_date
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

def train_model_task(task_id, parameters):
    """Background task to train the model"""
    try:
        # Update task status
        training_tasks[task_id]['status'] = 'running'
        
        # Fix for yfinance data
        # Monkey patch the yfinance download function to handle multi-index columns
        original_download = yf.download
        
        def patched_download(*args, **kwargs):
            df = original_download(*args, **kwargs)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        
        # Apply the monkey patch
        yf.download = patched_download
        
        # Initialize the trading system
        trading_system = AITradingSystem(
            ticker=parameters['ticker'],
            start_date=parameters['start_date'],
            end_date=parameters['end_date'],
            seq_length=int(parameters['seq_length']),
            verbose=True,
            use_continuous_action_space=parameters['use_continuous_action_space']
        )
        
        # Restore original function
        yf.download = original_download
        
        # Save the initial state for reference
        training_tasks[task_id]['initial_data'] = {
            'data_shape': trading_system.scaled_features.shape,
            'data_sample': trading_system.raw_data.head().to_dict()
        }
        
        # Train the model
        num_episodes = int(parameters['num_episodes'])
        best_reward = trading_system.train_dqn(
            num_episodes=num_episodes,
            double_dqn=True,
            dueling_dqn=True
        )
        
        # Save model and relevant data
        model_dir = os.path.join('static', 'models', task_id)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'model.keras')
        trading_system.dqn_model.save(model_path)
        
        # Save the scaler for later use
        scaler_path = os.path.join(model_dir, 'scaler.json')
        scaler_data = {
            "min_": trading_system.price_scaler.min_.tolist(),
            "scale_": trading_system.price_scaler.scale_.tolist()
        }
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f)
        
        # Run simulation
        simulation_results = simulate_trading(trading_system)
        
        # Create performance visualizations
        performance_plots = create_performance_plots(trading_system, simulation_results)
        
        # Store results
        trained_models[task_id] = {
            'trading_system': trading_system,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'simulation_results': simulation_results,
            'performance_plots': performance_plots,
            'parameters': parameters
        }
        
        # Update task status
        training_tasks[task_id]['status'] = 'completed'
        training_tasks[task_id]['results'] = {
            'best_reward': best_reward,
            'simulation_results': simulation_results,
            'performance_plots': performance_plots
        }
        
    except Exception as e:
        training_tasks[task_id]['status'] = 'failed'
        training_tasks[task_id]['error'] = str(e)
        print(f"Error in training task {task_id}: {e}")
        import traceback
        traceback.print_exc()

@app.route('/api/train', methods=['POST'])
def train_model():
    """Start a model training task"""
    data = request.get_json()
    
    # Generate a unique task ID
    task_id = f"task_{int(time.time())}"
    
    # Store task information
    training_tasks[task_id] = {
        'status': 'pending',
        'parameters': data,
        'created_at': datetime.now().isoformat()
    }
    
    # Start training in a background thread
    thread = threading.Thread(target=train_model_task, args=(task_id, data))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'success',
        'task_id': task_id,
        'message': 'Training started in the background'
    })

@app.route('/api/task_status/<task_id>')
def task_status(task_id):
    """Get the status of a training task"""
    if task_id not in training_tasks:
        return jsonify({
            'status': 'error',
            'message': 'Task not found'
        })
    
    task = training_tasks[task_id]
    response = {
        'status': task['status'],
        'created_at': task['created_at']
    }
    
    if task['status'] == 'completed':
        response['results'] = task['results']
    elif task['status'] == 'failed':
        response['error'] = task.get('error', 'Unknown error')
    
    return jsonify(response)

@app.route('/api/models')
def list_models():
    """List all trained models"""
    models = []
    for task_id, task in training_tasks.items():
        if task['status'] == 'completed':
            models.append({
                'task_id': task_id,
                'created_at': task['created_at'],
                'parameters': task['parameters'],
                'results_summary': {
                    'best_reward': task['results']['best_reward'],
                    'final_value': task['results']['simulation_results']['final_value'],
                    'gain_pct': task['results']['simulation_results']['gain_pct'],
                    'total_trades': task['results']['simulation_results']['total_trades']
                }
            })
    
    return jsonify({
        'status': 'success',
        'models': models
    })

def simulate_trading(trading_system):
    """Simulate trading with the trained DQN model"""
    state = trading_system.env.reset()
    done = False
    buy_count = sell_count = hold_count = 0
    initial_portfolio = trading_system.env.portfolio_value
    portfolio_history = [initial_portfolio]
    action_history = []
    price_history = []
    
    # Get the first price point
    first_price_scaled = trading_system.scaled_features[trading_system.seq_length, 0]
    first_price = trading_system.price_scaler.inverse_transform([[first_price_scaled]])[0, 0]
    price_history.append(first_price)
    
    while not done:
        state_tensor = tf.convert_to_tensor(state[np.newaxis, :], dtype=tf.float32)
        
        if trading_system.use_continuous_action_space:
            action_value = trading_system.dqn_model(state_tensor)[0, 0].numpy()
            if action_value < -0.05:
                action_type = 'sell'
                sell_count += 1
            elif action_value > 0.05:
                action_type = 'buy'
                buy_count += 1
            else:
                action_type = 'hold'
                hold_count += 1
            
            action_to_take = action_value
        else:
            q_values = trading_system.dqn_model(state_tensor)
            action = np.argmax(q_values[0])
            
            if action == 0:
                action_type = 'sell'
                sell_count += 1
            elif action == 1:
                action_type = 'hold'
                hold_count += 1
            else:
                action_type = 'buy'
                buy_count += 1
            
            action_to_take = action
        
        # Record the action
        action_history.append(action_type)
        
        # Take the step
        state, reward, done = trading_system.env.step(action_to_take)
        
        if not done:
            # Record portfolio value
            portfolio_history.append(trading_system.env.portfolio_value)
            
            # Get current price
            current_step = trading_system.env.current_step
            price_scaled = trading_system.scaled_features[current_step, 0]
            price = trading_system.price_scaler.inverse_transform([[price_scaled]])[0, 0]
            price_history.append(price)
    
    # Calculate final metrics
    final_portfolio = trading_system.env.portfolio_value
    performance_metrics = trading_system.env.get_performance_metrics()
    
    # Calculate buy-and-hold comparison
    last_price = price_history[-1]
    initial_btc = initial_portfolio / price_history[0]
    buyhold_value = initial_btc * last_price
    buyhold_comparison = (final_portfolio / buyhold_value - 1) * 100
    
    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "initial_value": initial_portfolio,
        "final_value": final_portfolio,
        "gain_pct": ((final_portfolio / initial_portfolio) - 1) * 100,
        "sharpe_ratio": performance_metrics['sharpe_ratio'],
        "max_drawdown": performance_metrics['max_drawdown'],
        "win_rate": performance_metrics['win_rate'],
        "total_trades": performance_metrics['num_trades'],
        "buyhold_value": buyhold_value,
        "vs_buyhold_pct": buyhold_comparison,
        "portfolio_history": portfolio_history,
        "action_history": action_history,
        "price_history": price_history
    }

def create_performance_plots(trading_system, simulation_results):
    """Create visualization plots for the trading performance"""
    # 1. Portfolio Value vs Price
    portfolio_df = pd.DataFrame({
        'step': range(len(simulation_results['portfolio_history'])),
        'portfolio_value': simulation_results['portfolio_history'],
        'price': simulation_results['price_history']
    })
    
    # Normalize values to start at 100 for easier comparison
    portfolio_df['portfolio_normalized'] = portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0] * 100
    portfolio_df['price_normalized'] = portfolio_df['price'] / portfolio_df['price'].iloc[0] * 100
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=portfolio_df['step'], 
        y=portfolio_df['portfolio_normalized'],
        mode='lines',
        name='AI Strategy'
    ))
    fig1.add_trace(go.Scatter(
        x=portfolio_df['step'],
        y=portfolio_df['price_normalized'],
        mode='lines',
        name='Buy & Hold'
    ))
    fig1.update_layout(
        title='Portfolio Performance vs Buy & Hold (Normalized to 100)',
        xaxis_title='Trading Steps',
        yaxis_title='Value (Starting = 100)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # 2. Trading Actions Visualization
    actions_df = pd.DataFrame({
        'step': range(len(simulation_results['action_history'])),
        'action': simulation_results['action_history'],
        'price': simulation_results['price_history']
    })
    
    # Create markers for buy/sell actions
    buy_points = actions_df[actions_df['action'] == 'buy']
    sell_points = actions_df[actions_df['action'] == 'sell']
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=actions_df['step'],
        y=actions_df['price'],
        mode='lines',
        name='Price'
    ))
    fig2.add_trace(go.Scatter(
        x=buy_points['step'],
        y=buy_points['price'],
        mode='markers',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        name='Buy'
    ))
    fig2.add_trace(go.Scatter(
        x=sell_points['step'],
        y=sell_points['price'],
        mode='markers',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        name='Sell'
    ))
    fig2.update_layout(
        title='Trading Actions on Price Chart',
        xaxis_title='Trading Steps',
        yaxis_title='Price',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # 3. Performance Metrics in a Pie Chart
    action_counts = {
        'Buy': simulation_results['buy_count'],
        'Hold': simulation_results['hold_count'],
        'Sell': simulation_results['sell_count']
    }
    
    fig3 = go.Figure(data=[go.Pie(
        labels=list(action_counts.keys()),
        values=list(action_counts.values()),
        hole=.3
    )])
    fig3.update_layout(title='Trading Action Distribution')
    
    # 4. Portfolio Growth Over Time
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=portfolio_df['step'],
        y=portfolio_df['portfolio_value'],
        mode='lines',
        name='Portfolio Value'
    ))
    fig4.update_layout(
        title='Portfolio Value Over Time',
        xaxis_title='Trading Steps',
        yaxis_title='Value ($)'
    )
    
    return {
        'performance_comparison': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'trading_actions': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'action_distribution': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder),
        'portfolio_growth': json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    }

@app.route('/api/model_details/<task_id>')
def model_details(task_id):
    """Get detailed information about a trained model"""
    if task_id not in training_tasks or training_tasks[task_id]['status'] != 'completed':
        return jsonify({
            'status': 'error',
            'message': 'Model not found or training not completed'
        })
    
    task = training_tasks[task_id]
    
    return jsonify({
        'status': 'success',
        'task_id': task_id,
        'parameters': task['parameters'],
        'results': task['results']
    })

@app.route('/api/load_pretrained_model')
def load_pretrained_model():
    """Load the pre-trained model and run a simulation with it"""
    global pretrained_model
    
    try:
        # Load the pre-trained model using absolute paths
        model_path = r"C:\Users\User\newest_ai\saved_models\1000_bitcoin_model.keras"
        scaler_path = r"C:\Users\User\newest_ai\saved_models\1000_scaler.json"
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return jsonify({
                'status': 'error',
                'message': f'Pre-trained model files not found at {model_path}'
            })
        
        # Initialize AITradingSystem with default parameters
        # but don't train a new model
        ticker = 'BTC-USD'
        start_date = (datetime.now().replace(year=datetime.now().year-4)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fix for yfinance data
        original_download = yf.download
        
        def patched_download(*args, **kwargs):
            df = original_download(*args, **kwargs)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        
        # Apply the monkey patch
        yf.download = patched_download
        
        # Create a modified version of AITradingSystem that loads the pre-trained model
        class PretrainedAITradingSystem(AITradingSystem):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Override the model creation methods to prevent training
                
            def create_models(self):
                """Override to load pre-trained model instead of creating new ones"""
                physical_devices = tf.config.list_physical_devices('GPU')
                if physical_devices:
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
                    print("Using GPU for inference")
                
                # Load scaler data
                with open(scaler_path, 'r') as f:
                    scaler_data = json.load(f)
                
                # Recreate scaler
                self.price_scaler = MinMaxScaler()
                self.price_scaler.min_ = np.array(scaler_data["min_"])
                self.price_scaler.scale_ = np.array(scaler_data["scale_"])
                
                # Load pre-trained models
                self.dqn_model = tf.keras.models.load_model(model_path)
                print(f"Loaded pre-trained model from {model_path}")
                
                # Create dummy models for compatibility
                self.predictive_model = self.dqn_model  # Just for compatibility
                self.lstm_model = self.dqn_model  # Just for compatibility
        
        # Initialize with the pre-trained model
        trading_system = PretrainedAITradingSystem(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            seq_length=60,
            verbose=True,
            use_continuous_action_space=False
        )
        
        # Restore original function
        yf.download = original_download
        
        # Run simulation
        simulation_results = simulate_trading(trading_system)
        
        # Create performance visualizations
        performance_plots = create_performance_plots(trading_system, simulation_results)
        
        # Store as pretrained model
        pretrained_model = {
            'trading_system': trading_system,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'simulation_results': simulation_results,
            'performance_plots': performance_plots,
            'parameters': {
                'ticker': ticker,
                'start_date': start_date,
                'end_date': end_date,
                'seq_length': 60,
                'use_continuous_action_space': False,
                'pretrained': True
            }
        }
        
        # Store in trained_models with a special ID
        task_id = 'pretrained_model'
        trained_models[task_id] = pretrained_model
        
        # Create a fake task for UI consistency
        training_tasks[task_id] = {
            'status': 'completed',
            'parameters': pretrained_model['parameters'],
            'created_at': datetime.now().isoformat(),
            'results': {
                'best_reward': 0,  # We don't have this info for pre-trained model
                'simulation_results': simulation_results,
                'performance_plots': performance_plots
            }
        }
        
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'message': 'Pre-trained model loaded successfully',
            'simulation_results': simulation_results,
            'performance_plots': performance_plots
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)