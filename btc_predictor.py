"""
Fixed BTC Predictor with Sentiment Analysis
Cleaned and improved error handling
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
import requests
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def download_btc_data(days_back=3650):
    """Download BTC historical data from CoinGecko"""
    print("📥 Downloading BTC historical data...")
    
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': str(days_back),
            'interval': 'daily'
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Create OHLC data
        df['Open'] = df['Close'].shift(1).fillna(df['Close'])
        df['High'] = df[['Open', 'Close']].max(axis=1) * 1.005
        df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.995
        
        df = df[['Open', 'High', 'Low', 'Close']]
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        save_path = 'data/btc_price_data.csv'
        df.to_csv(save_path)
        
        print(f"✓ Downloaded {len(df)} days of BTC data")
        print(f"✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"✓ Saved to: {save_path}")
        
        return df
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        raise


def load_btc_data():
    """Load BTC data from CSV or download if not exists"""
    data_path = 'data/btc_price_data.csv'
    
    if os.path.exists(data_path):
        print(f"📂 Loading BTC data from: {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(df)} days of data")
        print(f"✓ Date range: {df.index[0].date()} to {df.index[-1].date()}")
        
        # Check if data is old
        last_date = pd.to_datetime(df.index[-1])
        days_old = (datetime.now() - last_date).days
        
        if days_old > 7:
            print(f"⚠ Data is {days_old} days old. Updating...")
            return download_btc_data(days_back=300)  # Changed from 3650 to 300
        
        return df
    else:
        print("📥 No local data found. Downloading...")
        return download_btc_data(days_back=300)  # Changed from 3650 to 300


def load_sentiment_data():
    """Load sentiment data from CSV"""
    print("\n📊 Loading sentiment data...")
    
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print("⚠ No data directory found")
        return None
    
    files = [f for f in os.listdir(data_dir) if f.startswith('sentiment_daily_')]
    
    if not files:
        print("⚠ No sentiment data found")
        print("Run sentiment_collector.py and sentiment_analyzer.py first!")
        return None
    
    latest_file = sorted(files)[-1]
    filepath = os.path.join(data_dir, latest_file)
    
    print(f"Loading: {filepath}")
    df_sent = pd.read_csv(filepath)
    df_sent['date'] = pd.to_datetime(df_sent['date'])
    df_sent.set_index('date', inplace=True)
    
    print(f"✓ Loaded sentiment data: {len(df_sent)} days")
    print(f"✓ Date range: {df_sent.index[0].date()} to {df_sent.index[-1].date()}")
    
    return df_sent


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def create_technical_indicators(df):
    """Create technical indicators"""
    print("\n📊 Creating technical indicators...")
    
    df_ti = df.copy()
    
    # Moving Averages
    df_ti['SMA_20'] = df_ti["Close"].rolling(window=20).mean()
    df_ti['EMA_20'] = df_ti["Close"].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df_ti["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_ti['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df_ti["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df_ti["Close"].ewm(span=26, adjust=False).mean()
    df_ti["MACD"] = exp12 - exp26
    df_ti["Signal_Line"] = df_ti["MACD"].ewm(span=9, adjust=False).mean()
    
    # ATR
    high_low = df_ti['High'] - df_ti['Low']
    high_prev_close = np.abs(df_ti['High'] - df_ti['Close'].shift(1))
    low_prev_close = np.abs(df_ti['Low'] - df_ti['Close'].shift(1))
    df_ti['TR'] = np.maximum.reduce([high_low, high_prev_close, low_prev_close])
    df_ti['ATR'] = df_ti['TR'].ewm(span=14, adjust=False).mean()
    
    # Volatility
    df_ti['Volatility'] = df_ti['Close'].rolling(window=10).std()
    
    # Drop NaN
    result = df_ti.dropna()
    print(f"✓ Created indicators, {len(result)} samples remaining")
    
    return result


def merge_sentiment_with_price(df_price, df_sentiment):
    """Merge price and sentiment data"""
    if df_sentiment is None:
        print("⚠ No sentiment data - running baseline model only")
        return df_price, []
    
    print("\n🔗 Merging sentiment with price data...")
    
    # Ensure both indices are datetime
    df_price.index = pd.to_datetime(df_price.index)
    df_sentiment.index = pd.to_datetime(df_sentiment.index)
    
    # Check date overlap
    price_start = df_price.index.min()
    price_end = df_price.index.max()
    sent_start = df_sentiment.index.min()
    sent_end = df_sentiment.index.max()
    
    print(f"\n📅 Date Range Analysis:")
    print(f"  Price data:     {price_start.date()} to {price_end.date()}")
    print(f"  Sentiment data: {sent_start.date()} to {sent_end.date()}")
    
    overlap_start = max(price_start, sent_start)
    overlap_end = min(price_end, sent_end)
    
    if overlap_start > overlap_end:
        print("\n⚠ Warning: No date overlap between datasets!")
        print("Using left join and forward fill for missing sentiment...")
        df_merged = df_price.join(df_sentiment, how='left')
        
        # Forward fill sentiment values
        sentiment_cols = [col for col in df_sentiment.columns if 'sentiment' in col.lower()]
        for col in sentiment_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    else:
        print(f"  Overlap period: {overlap_start.date()} to {overlap_end.date()}")
        # Inner join to use only overlapping dates
        df_merged = df_price.join(df_sentiment, how='inner')
    
    print(f"✓ After merge: {len(df_merged)} samples")
    
    # Create sentiment lag features
    sentiment_cols = []
    if 'sentiment_mean' in df_merged.columns:
        df_merged['sentiment_lag1'] = df_merged['sentiment_mean'].shift(1)
        df_merged['sentiment_lag3'] = df_merged['sentiment_mean'].shift(3)
        df_merged['sentiment_lag7'] = df_merged['sentiment_mean'].shift(7)
        
        sentiment_cols = ['sentiment_mean', 'sentiment_std', 'sentiment_pos_mean', 
                         'sentiment_neg_mean', 'sentiment_momentum',
                         'sentiment_lag1', 'sentiment_lag3', 'sentiment_lag7']
        
        # Keep only columns that exist
        sentiment_cols = [col for col in sentiment_cols if col in df_merged.columns]
    
    df_merged = df_merged.dropna()
    
    print(f"✓ After creating lag features: {len(df_merged)} samples")
    print(f"✓ Available sentiment features: {len(sentiment_cols)}")
    
    return df_merged, sentiment_cols


def create_features_and_targets(df, sentiment_features=[]):
    """Create feature matrix and target"""
    print("\n🎯 Creating features and targets...")
    
    df_feat = df.copy()
    
    # Normalize Close price
    scaler = MinMaxScaler()
    df_feat['Close_Normalized'] = scaler.fit_transform(df_feat[['Close']])
    
    # Price lag features
    df_feat["Lag_1D"] = df_feat["Close"].shift(1)
    df_feat["Lag_7D"] = df_feat["Close"].shift(7)
    df_feat["Lag_30D"] = df_feat["Close"].shift(30)
    
    # Target: next day's price
    df_feat["Target_Price"] = df_feat["Close"].shift(-1)
    
    # Remove last row (no target) and any NaN
    df_feat = df_feat.iloc[:-1].dropna()
    
    print(f"✓ After lag features: {len(df_feat)} samples")
    
    # Base technical features
    base_features = [
        'Close_Normalized', 'Lag_1D', 'Lag_7D', 'Lag_30D',
        'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Signal_Line',
        'ATR', 'Volatility'
    ]
    
    # Combine with sentiment
    all_features = base_features + sentiment_features
    available_features = [f for f in all_features if f in df_feat.columns]
    
    X = df_feat[available_features]
    Y = df_feat['Target_Price']
    
    print(f"✓ Features: {len(available_features)} ({len([f for f in available_features if f in base_features])} technical + {len([f for f in available_features if f in sentiment_features])} sentiment)")
    print(f"✓ Final samples: {len(X)}")
    
    if len(X) == 0:
        raise ValueError("Feature matrix is empty! Check your data.")
    
    return X, Y, df_feat.index, available_features


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_and_evaluate(X_train, Y_train, X_test, Y_test, alpha=10000):
    """Train and evaluate Ridge regression model"""
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, Y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = {
        'MAE': mean_absolute_error(Y_train, y_train_pred),
        'RMSE': np.sqrt(mean_squared_error(Y_train, y_train_pred)),
        'R2': r2_score(Y_train, y_train_pred)
    }
    
    test_metrics = {
        'MAE': mean_absolute_error(Y_test, y_test_pred),
        'RMSE': np.sqrt(mean_squared_error(Y_test, y_test_pred)),
        'R2': r2_score(Y_test, y_test_pred)
    }
    
    return model, train_metrics, test_metrics, y_test_pred


def compare_models(baseline_metrics, enhanced_metrics):
    """Compare baseline vs enhanced model"""
    print("\n" + "="*70)
    print("MODEL COMPARISON: Baseline vs Sentiment-Enhanced")
    print("="*70)
    
    for metric in ['MAE', 'RMSE', 'R2']:
        baseline_val = baseline_metrics[metric]
        enhanced_val = enhanced_metrics[metric]
        
        if metric == 'R2':
            improvement = ((enhanced_val - baseline_val) / abs(baseline_val)) * 100
            better = enhanced_val > baseline_val
        else:
            improvement = ((baseline_val - enhanced_val) / baseline_val) * 100
            better = enhanced_val < baseline_val
        
        symbol = "✓" if better else "✗"
        
        print(f"\n{metric}:")
        print(f"  Baseline:  {baseline_val:.4f}")
        print(f"  Enhanced:  {enhanced_val:.4f}")
        print(f"  Change:    {improvement:+.2f}% {symbol}")


def plot_comparison(dates_test, Y_test, baseline_pred, enhanced_pred):
    """Plot model comparison"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Baseline plot
    axes[0].plot(dates_test, Y_test.values, label='Actual Price', 
                color='blue', linewidth=1.5)
    axes[0].plot(dates_test, baseline_pred, label='Baseline Prediction',
                color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].set_title('Baseline Model (Technical Indicators Only)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Enhanced plot
    axes[1].plot(dates_test, Y_test.values, label='Actual Price',
                color='blue', linewidth=1.5)
    axes[1].plot(dates_test, enhanced_pred, label='Enhanced Prediction (with Sentiment)',
                color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].set_title('Enhanced Model (Technical + Sentiment)',
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Price (USD)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('data/figures', exist_ok=True)
    plt.savefig('data/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n💾 Comparison plot saved: data/figures/model_comparison.png")
    plt.show()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main execution"""
    print("="*70)
    print("BTC PREDICTOR WITH SENTIMENT ANALYSIS")
    print("="*70)
    
    TEST_SIZE = 0.15
    ALPHA = 10000
    
    try:
        # Step 1: Load BTC data
        print("\n[1/6] Loading BTC price data...")
        btc_df = load_btc_data()
        
        # Step 2: Create technical indicators
        print("\n[2/6] Creating technical indicators...")
        btc_df = create_technical_indicators(btc_df)
        
        # Step 3: Load sentiment
        print("\n[3/6] Loading sentiment data...")
        df_sentiment = load_sentiment_data()
        
        # Step 4: Train baseline model
        print("\n[4/6] Training BASELINE model...")
        X_base, Y_base, dates_base, _ = create_features_and_targets(btc_df)
        
        X_train_base, X_test_base, Y_train_base, Y_test_base, _, dates_test = train_test_split(
            X_base, Y_base, dates_base, test_size=TEST_SIZE, shuffle=False
        )
        
        baseline_model, baseline_train, baseline_test, baseline_pred = train_and_evaluate(
            X_train_base, Y_train_base, X_test_base, Y_test_base, ALPHA
        )
        
        print("\n--- Baseline Model Results ---")
        print(f"Test MAE:  ${baseline_test['MAE']:.2f}")
        print(f"Test RMSE: ${baseline_test['RMSE']:.2f}")
        print(f"Test R²:   {baseline_test['R2']:.4f}")
        
        # Step 5: Train enhanced model (if sentiment available)
        if df_sentiment is not None:
            print("\n[5/6] Training ENHANCED model...")
            btc_merged, sentiment_cols = merge_sentiment_with_price(btc_df, df_sentiment)
            
            if btc_merged is None or len(btc_merged) < 50:
                print("\n⚠ Not enough data for enhanced model")
                print("[6/6] Skipped - comparison not possible")
                return
            
            X_enh, Y_enh, dates_enh, features_used = create_features_and_targets(
                btc_merged, sentiment_features=sentiment_cols
            )
            
            X_train_enh, X_test_enh, Y_train_enh, Y_test_enh, _, dates_test_enh = train_test_split(
                X_enh, Y_enh, dates_enh, test_size=TEST_SIZE, shuffle=False
            )
            
            enhanced_model, enhanced_train, enhanced_test, enhanced_pred = train_and_evaluate(
                X_train_enh, Y_train_enh, X_test_enh, Y_test_enh, ALPHA
            )
            
            print("\n--- Enhanced Model Results ---")
            print(f"Test MAE:  ${enhanced_test['MAE']:.2f}")
            print(f"Test RMSE: ${enhanced_test['RMSE']:.2f}")
            print(f"Test R²:   {enhanced_test['R2']:.4f}")
            
            # Step 6: Compare
            print("\n[6/6] Comparing models...")
            compare_models(baseline_test, enhanced_test)
            
            # Plot
            plot_comparison(dates_test_enh, Y_test_enh,
                          baseline_model.predict(X_test_enh), enhanced_pred)
            
            print("\n" + "="*70)
            print("✅ ANALYSIS COMPLETE!")
            print("="*70)
            
            improvement = ((baseline_test['MAE'] - enhanced_test['MAE']) / 
                          baseline_test['MAE'] * 100)
            print(f"\n🎯 Sentiment improved prediction by: {improvement:+.2f}%")
            
        else:
            print("\n[5/6] Skipped - no sentiment data")
            print("[6/6] Skipped - no comparison possible")
            print("\n⚠ Run sentiment_collector.py and sentiment_analyzer.py first")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()