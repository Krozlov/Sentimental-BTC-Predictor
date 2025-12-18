"""
Fixed BTC Predictor with Sentiment Analysis
Added debugging and error handling for empty datasets
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                            r2_score, mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split
import os
import requests
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


def downloadAndSaveBTCData(days_back=3650):
    """One-time download of BTC data and save to CSV"""
    print("📥 Downloading BTC historical data (one-time setup)...")
    
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
        
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        df['Open'] = df['Close'].shift(1).fillna(df['Close'])
        df['High'] = df[['Open', 'Close']].max(axis=1) * 1.005
        df['Low'] = df[['Open', 'Close']].min(axis=1) * 0.995
        
        df = df[['Open', 'High', 'Low', 'Close']]
        
        os.makedirs('data', exist_ok=True)
        save_path = 'data/btc_price_data.csv'
        df.to_csv(save_path)
        
        print(f"✓ Downloaded {len(df)} days of BTC data")
        print(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
        print(f"✓ Saved to: {save_path}")
        
        return df
        
    except Exception as e:
        print(f"✗ CoinGecko download failed: {e}")
        print("\nTrying alternative source...")
        
        try:
            return downloadFromBinance(days_back)
        except:
            raise ValueError("Failed to download BTC data. Please check internet connection.")


def downloadFromBinance(days_back):
    """Backup method using Binance API"""
    print("Trying Binance API...")
    
    url = "https://api.binance.com/api/v3/klines"
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
    
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1d',
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }
    
    all_data = []
    
    while start_time < end_time:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            break
        
        all_data.extend(data)
        start_time = data[-1][0] + 1
        params['startTime'] = start_time
    
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col])
    
    df = df[['Open', 'High', 'Low', 'Close']]
    
    save_path = 'data/btc_price_data.csv'
    df.to_csv(save_path)
    
    print(f"✓ Downloaded {len(df)} days from Binance")
    print(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
    print(f"✓ Saved to: {save_path}")
    
    return df


def loadBTCData():
    """Load BTC data from CSV or download if not exists"""
    data_path = 'data/btc_price_data.csv'
    
    if os.path.exists(data_path):
        print(f"📂 Loading BTC data from: {data_path}")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"✓ Loaded {len(df)} days of data")
        print(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
        
        last_date = pd.to_datetime(df.index[-1])
        days_old = (datetime.now() - last_date).days
        
        if days_old > 7:
            print(f"⚠ Data is {days_old} days old. Updating...")
            return downloadAndSaveBTCData()
        
        return df
    else:
        print("📥 No local data found. Downloading...")
        return downloadAndSaveBTCData()


def preprocessData(df):
    """Preprocess BTC price data"""
    print("\n🔧 Preprocessing data...")
    
    df = df.copy()
    
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Close_Normalized'] = scaler.fit_transform(df[['Close']])
    
    print(f"✓ Preprocessed {len(df)} samples")
    return df


def createTechnicalIndicators(df):
    """Create technical indicators"""
    print("📊 Creating technical indicators...")
    
    df_ti = df.copy()
    
    df_ti['SMA_20'] = df_ti["Close"].rolling(window=20).mean()
    df_ti['EMA_20'] = df_ti["Close"].ewm(span=20, adjust=False).mean()
    
    delta = df_ti["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_ti['RSI'] = 100 - (100 / (1 + rs))
    
    exp12 = df_ti["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df_ti["Close"].ewm(span=26, adjust=False).mean()
    df_ti["MACD"] = exp12 - exp26
    df_ti["Signal_Line"] = df_ti["MACD"].ewm(span=9, adjust=False).mean()
    df_ti["MACD_Hist"] = df_ti["MACD"] - df_ti["Signal_Line"]
    
    high_low = df_ti['High'] - df_ti['Low']
    high_prev_close = np.abs(df_ti['High'] - df_ti['Close'].shift(1))
    low_prev_close = np.abs(df_ti['Low'] - df_ti['Close'].shift(1))
    df_ti['TR'] = np.maximum.reduce([high_low, high_prev_close, low_prev_close])
    df_ti['ATR_14'] = df_ti['TR'].ewm(span=14, adjust=False).mean()
    
    df_ti['Rolling_Std_10'] = df_ti['Close'].rolling(window=10).std()
    
    result = df_ti.dropna()
    print(f"✓ Created indicators, {len(result)} samples remaining")
    
    return result


def loadSentimentData():
    """Load sentiment data from CSV"""
    print("\n📊 Loading sentiment data...")
    
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print("⚠ No data directory found")
        return None
    
    files = [f for f in os.listdir(data_dir) if f.startswith('sentiment_daily_')]
    
    if not files:
        print("⚠ No sentiment data found")
        print("Run sentiment_analyzer.py first!")
        return None
    
    latest_file = sorted(files)[-1]
    filepath = os.path.join(data_dir, latest_file)
    
    print(f"Loading: {filepath}")
    df_sent = pd.read_csv(filepath)
    df_sent['date'] = pd.to_datetime(df_sent['date'])
    df_sent.set_index('date', inplace=True)
    
    print(f"✓ Loaded sentiment data: {len(df_sent)} days")
    print(f"✓ Date range: {df_sent.index[0]} to {df_sent.index[-1]}")
    
    return df_sent


def mergeSentimentWithPrice(df_price, df_sentiment):
    """Merge price and sentiment data with improved debugging"""
    if df_sentiment is None:
        print("⚠ No sentiment data - running baseline model only")
        return df_price, []
    
    print("\n🔗 Merging sentiment with price data...")
    
    # Ensure both indices are datetime
    df_price.index = pd.to_datetime(df_price.index)
    df_sentiment.index = pd.to_datetime(df_sentiment.index)
    
    # DEBUG: Check date overlap
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
        print("\n❌ ERROR: No date overlap between price and sentiment data!")
        print("Please ensure sentiment data covers the same time period as price data.")
        return None, []
    
    print(f"  Overlap period: {overlap_start.date()} to {overlap_end.date()}")
    
    # Merge
    df_merged = df_price.join(df_sentiment, how='inner')  # Changed to inner join
    
    print(f"✓ After merge: {len(df_merged)} samples")
    
    if len(df_merged) == 0:
        print("\n❌ ERROR: Merge resulted in 0 samples!")
        print("Trying with outer join and forward fill...")
        
        df_merged = df_price.join(df_sentiment, how='left')
        
        sentiment_cols = ['sentiment_mean', 'sentiment_std', 'sentiment_pos_mean', 
                         'sentiment_neg_mean', 'sentiment_momentum']
        
        for col in sentiment_cols:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"✓ After forward fill: {len(df_merged)} samples")
    
    # Create sentiment lag features
    sentiment_cols = ['sentiment_mean', 'sentiment_std', 'sentiment_pos_mean', 
                     'sentiment_neg_mean', 'sentiment_momentum']
    
    available_sentiment_cols = [col for col in sentiment_cols if col in df_merged.columns]
    
    if 'sentiment_mean' in df_merged.columns:
        df_merged['sentiment_lag1'] = df_merged['sentiment_mean'].shift(1)
        df_merged['sentiment_lag3'] = df_merged['sentiment_mean'].shift(3)
        df_merged['sentiment_lag7'] = df_merged['sentiment_mean'].shift(7)
        available_sentiment_cols.extend(['sentiment_lag1', 'sentiment_lag3', 'sentiment_lag7'])
    
    df_merged = df_merged.dropna()
    
    print(f"✓ After creating lag features: {len(df_merged)} samples")
    print(f"✓ Available sentiment features: {len(available_sentiment_cols)}")
    
    return df_merged, available_sentiment_cols


def createFeaturesAndTargets(df, sentiment_features=[]):
    """Create feature matrix and target"""
    print("\n🎯 Creating features and targets...")
    
    df_feat = df.copy()
    
    # Price lag features
    df_feat["Lag_1D"] = df_feat["Close"].shift(1)
    df_feat["Lag_7D"] = df_feat["Close"].shift(7)
    df_feat["Lag_30D"] = df_feat["Close"].shift(30)
    
    # Target: next day's price
    df_feat["Target_Price"] = df_feat["Close"].shift(-1)
    
    df_feat = df_feat.iloc[:-1].dropna()
    
    print(f"✓ After lag features: {len(df_feat)} samples")
    
    # Base features
    base_features = ['Close_Normalized', 'Lag_1D', 'Lag_7D', 'Lag_30D',
                    'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Signal_Line',
                    'MACD_Hist', 'ATR_14', 'Rolling_Std_10']
    
    # Combine with sentiment
    all_features = base_features + sentiment_features
    available_features = [f for f in all_features if f in df_feat.columns]
    
    X = df_feat[available_features]
    Y = df_feat['Target_Price']
    
    print(f"✓ Features: {len(available_features)} ({len(base_features)} technical + {len(sentiment_features)} sentiment)")
    print(f"✓ Final samples: {len(X)}")
    
    if len(X) == 0:
        print("\n❌ ERROR: Feature matrix is empty!")
        print("Available columns:", df_feat.columns.tolist())
    
    return X, Y, df_feat.index, available_features


def trainAndEvaluate(X_train, Y_train, X_test, Y_test, alpha=10000):
    """Train and evaluate Ridge regression model"""
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, Y_train)
    
    y_train_pred = model.predict(X_train)
    train_metrics = {
        'Train_MAE': mean_absolute_error(Y_train, y_train_pred),
        'Train_RMSE': np.sqrt(mean_squared_error(Y_train, y_train_pred)),
        'Train_R2': r2_score(Y_train, y_train_pred),
        'Train_MAPE': mean_absolute_percentage_error(Y_train, y_train_pred) * 100
    }
    
    y_test_pred = model.predict(X_test)
    test_metrics = {
        'Test_MAE': mean_absolute_error(Y_test, y_test_pred),
        'Test_RMSE': np.sqrt(mean_squared_error(Y_test, y_test_pred)),
        'Test_R2': r2_score(Y_test, y_test_pred),
        'Test_MAPE': mean_absolute_percentage_error(Y_test, y_test_pred) * 100
    }
    
    return model, {**train_metrics, **test_metrics}, y_test_pred


def compareModels(baseline_metrics, enhanced_metrics):
    """Compare baseline vs enhanced model"""
    print("\n" + "="*70)
    print("MODEL COMPARISON: Baseline vs Sentiment-Enhanced")
    print("="*70)
    
    metrics = ['Test_MAE', 'Test_RMSE', 'Test_R2', 'Test_MAPE']
    
    for metric in metrics:
        baseline_val = baseline_metrics[metric]
        enhanced_val = enhanced_metrics[metric]
        
        if metric == 'Test_R2':
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


def plotComparison(dates_test, Y_test, baseline_pred, enhanced_pred):
    """Plot model comparison"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    ax1.plot(dates_test, Y_test.values, label='Actual Price', 
            color='blue', linewidth=1)
    ax1.plot(dates_test, baseline_pred, label='Baseline Prediction',
            color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_title('Baseline Model (Technical Indicators Only)', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(dates_test, Y_test.values, label='Actual Price',
            color='blue', linewidth=1)
    ax2.plot(dates_test, enhanced_pred, label='Enhanced Prediction (with Sentiment)',
            color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_title('Enhanced Model (Technical + Sentiment)',
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price (USD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('data/figures', exist_ok=True)
    plt.savefig('data/figures/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n💾 Comparison plot saved: data/figures/model_comparison.png")
    plt.show()


def main():
    """Main execution"""
    print("="*70)
    print("STANDALONE BTC PREDICTOR WITH SENTIMENT ANALYSIS")
    print("="*70)
    
    TEST_SIZE = 0.15
    ALPHA = 10000
    
    # Step 1: Load BTC data
    print("\n[1/7] Loading BTC price data...")
    btc_df = loadBTCData()
    
    # Step 2: Preprocess
    print("\n[2/7] Preprocessing...")
    btc_df = preprocessData(btc_df)
    
    # Step 3: Technical indicators
    print("\n[3/7] Creating technical indicators...")
    btc_df = createTechnicalIndicators(btc_df)
    
    # Step 4: Load sentiment
    print("\n[4/7] Loading sentiment data...")
    df_sentiment = loadSentimentData()
    
    # Step 5: Train baseline model
    print("\n[5/7] Training BASELINE model...")
    X_base, Y_base, dates_base, _ = createFeaturesAndTargets(btc_df)
    
    if len(X_base) == 0:
        print("\n❌ Cannot proceed: No samples available for baseline model")
        return
    
    X_train_base, X_test_base, Y_train_base, Y_test_base, dates_train, dates_test = train_test_split(
        X_base, Y_base, dates_base, test_size=TEST_SIZE, shuffle=False
    )
    
    baseline_model, baseline_metrics, baseline_pred = trainAndEvaluate(
        X_train_base, Y_train_base, X_test_base, Y_test_base, ALPHA
    )
    
    print("\n--- Baseline Model Results ---")
    print(f"Test MAE:  ${baseline_metrics['Test_MAE']:.2f}")
    print(f"Test RMSE: ${baseline_metrics['Test_RMSE']:.2f}")
    print(f"Test R²:   {baseline_metrics['Test_R2']:.4f}")
    print(f"Test MAPE: {baseline_metrics['Test_MAPE']:.2f}%")
    
    # Step 6: Train enhanced model
    if df_sentiment is not None:
        print("\n[6/7] Training ENHANCED model...")
        btc_merged, sentiment_cols = mergeSentimentWithPrice(btc_df, df_sentiment)
        
        if btc_merged is None or len(btc_merged) == 0:
            print("\n❌ Cannot train enhanced model: Merge failed or resulted in empty dataset")
            print("[7/7] Skipped - no comparison possible")
            print("\n" + "="*70)
            print("✅ BASELINE MODEL COMPLETE (Enhanced model failed)")
            print("="*70)
            return
        
        X_enh, Y_enh, dates_enh, features_used = createFeaturesAndTargets(
            btc_merged, sentiment_features=sentiment_cols
        )
        
        if len(X_enh) == 0:
            print("\n❌ Cannot train enhanced model: Feature creation resulted in empty dataset")
            print("[7/7] Skipped - no comparison possible")
            return
        
        X_train_enh, X_test_enh, Y_train_enh, Y_test_enh, _, dates_test_enh = train_test_split(
            X_enh, Y_enh, dates_enh, test_size=TEST_SIZE, shuffle=False
        )
        
        enhanced_model, enhanced_metrics, enhanced_pred = trainAndEvaluate(
            X_train_enh, Y_train_enh, X_test_enh, Y_test_enh, ALPHA
        )
        
        print("\n--- Enhanced Model Results ---")
        print(f"Test MAE:  ${enhanced_metrics['Test_MAE']:.2f}")
        print(f"Test RMSE: ${enhanced_metrics['Test_RMSE']:.2f}")
        print(f"Test R²:   {enhanced_metrics['Test_R2']:.4f}")
        print(f"Test MAPE: {enhanced_metrics['Test_MAPE']:.2f}%")
        
        # Step 7: Compare
        print("\n[7/7] Comparing models...")
        compareModels(baseline_metrics, enhanced_metrics)
        
        # Plot
        plotComparison(dates_test_enh, Y_test_enh,
                      baseline_model.predict(X_test_enh), enhanced_pred)
        
    else:
        print("\n[6/7] Skipped - no sentiment data")
        print("[7/7] Skipped - no comparison possible")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    
    if df_sentiment is not None and 'enhanced_metrics' in locals():
        print("\n🎯 Results Summary:")
        print(f"  • Baseline Test MAE: ${baseline_metrics['Test_MAE']:.2f}")
        print(f"  • Enhanced Test MAE: ${enhanced_metrics['Test_MAE']:.2f}")
        
        improvement = ((baseline_metrics['Test_MAE'] - enhanced_metrics['Test_MAE']) / 
                      baseline_metrics['Test_MAE'] * 100)
        print(f"  • Improvement: {improvement:+.2f}%")
        
        print("\n📊 Next step: Run visualizations.py for more charts!")
    else:
        print("\n⚠ Run sentiment_collector.py and sentiment_analyzer.py first")
        print("Then re-run this script to compare models!")


if __name__ == "__main__":
    main()