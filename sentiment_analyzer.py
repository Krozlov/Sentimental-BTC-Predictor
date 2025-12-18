"""
Sentiment Analyzer for Bitcoin Text Data
Processes Reddit/News data and generates sentiment scores
Run this SECOND after collecting data
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from datetime import datetime
import os

class SentimentAnalyzer:
    def __init__(self):
        """Initialize VADER sentiment analyzer"""
        self.analyzer = SentimentIntensityAnalyzer()
        print("✓ VADER Sentiment Analyzer initialized")
    
    def clean_text(self, text):
        """Clean text for better sentiment analysis"""
        if pd.isna(text) or not text:
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s!?.,]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def analyze_text(self, text):
        """
        Analyze sentiment of text using VADER
        Returns: dict with compound, positive, negative, neutral scores
        """
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        
        scores = self.analyzer.polarity_scores(cleaned)
        return scores
    
    def process_dataframe(self, df):
        """
        Process entire dataframe and add sentiment scores
        
        Args:
            df: DataFrame with 'text' and 'date' columns
        """
        print("\n🔍 Analyzing sentiment...")
        
        # Combine title and text for better context
        if 'title' in df.columns:
            df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        else:
            df['full_text'] = df['text'].fillna('')
        
        # Analyze each text
        sentiments = []
        for idx, text in enumerate(df['full_text']):
            if idx % 100 == 0:
                print(f"  Processed {idx}/{len(df)} texts...")
            
            sentiment = self.analyze_text(text)
            sentiments.append(sentiment)
        
        # Add sentiment columns
        df['sentiment_compound'] = [s['compound'] for s in sentiments]
        df['sentiment_pos'] = [s['pos'] for s in sentiments]
        df['sentiment_neg'] = [s['neg'] for s in sentiments]
        df['sentiment_neu'] = [s['neu'] for s in sentiments]
        
        # Categorize sentiment
        df['sentiment_label'] = df['sentiment_compound'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        
        print(f"✓ Sentiment analysis complete!")
        return df
    
    def aggregate_daily_sentiment(self, df):
        """
        Aggregate sentiment scores by day
        This is what you'll merge with BTC price data
        """
        print("\n📊 Aggregating daily sentiment scores...")
        
        # Convert date to datetime if needed
        if df['date'].dtype != 'datetime64[ns]':
            df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and source type
        daily_agg = df.groupby('date').agg({
            'sentiment_compound': ['mean', 'std', 'min', 'max', 'count'],
            'sentiment_pos': 'mean',
            'sentiment_neg': 'mean',
            'sentiment_neu': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = ['date', 'sentiment_mean', 'sentiment_std', 
                            'sentiment_min', 'sentiment_max', 'text_count',
                            'sentiment_pos_mean', 'sentiment_neg_mean', 
                            'sentiment_neu_mean']
        
        # Fill NaN std with 0 (for days with single data point)
        daily_agg['sentiment_std'] = daily_agg['sentiment_std'].fillna(0)
        
        # Calculate sentiment momentum (change from previous day)
        daily_agg = daily_agg.sort_values('date')
        daily_agg['sentiment_momentum'] = daily_agg['sentiment_mean'].diff()
        daily_agg['sentiment_momentum'] = daily_agg['sentiment_momentum'].fillna(0)
        
        print(f"✓ Created daily aggregations for {len(daily_agg)} days")
        return daily_agg
    
    def create_source_specific_features(self, df):
        """Create separate features for Reddit vs News sentiment"""
        print("\n🔀 Creating source-specific features...")
        
        # Convert date to datetime
        if df['date'].dtype != 'datetime64[ns]':
            df['date'] = pd.to_datetime(df['date'])
        
        # Separate Reddit and News
        reddit_mask = df['source'].str.contains('Reddit', case=False, na=False)
        news_mask = df['source'].str.contains('News', case=False, na=False)
        
        # Aggregate Reddit sentiment
        reddit_daily = df[reddit_mask].groupby('date').agg({
            'sentiment_compound': 'mean'
        }).rename(columns={'sentiment_compound': 'reddit_sentiment'})
        
        # Aggregate News sentiment
        news_daily = df[news_mask].groupby('date').agg({
            'sentiment_compound': 'mean'
        }).rename(columns={'sentiment_compound': 'news_sentiment'})
        
        # Combine
        source_features = reddit_daily.join(news_daily, how='outer')
        source_features = source_features.fillna(0)  # Fill days with no data
        
        print(f"✓ Source-specific features created")
        return source_features.reset_index()
    
    def save_results(self, df_with_sentiment, daily_agg, source_features, output_dir='data'):
        """Save processed sentiment data"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed sentiment data
        detailed_path = f'{output_dir}/sentiment_detailed_{timestamp}.csv'
        df_with_sentiment.to_csv(detailed_path, index=False)
        print(f"\n💾 Detailed sentiment saved: {detailed_path}")
        
        # Save daily aggregated data (THIS IS WHAT YOU NEED FOR MODELING)
        daily_path = f'{output_dir}/sentiment_daily_{timestamp}.csv'
        daily_agg.to_csv(daily_path, index=False)
        print(f"💾 Daily sentiment saved: {daily_path}")
        
        # Save source-specific features
        source_path = f'{output_dir}/sentiment_by_source_{timestamp}.csv'
        source_features.to_csv(source_path, index=False)
        print(f"💾 Source-specific sentiment saved: {source_path}")
        
        return daily_path
    
    def print_summary_stats(self, df):
        """Print summary statistics"""
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*60)
        
        total = len(df)
        positive = len(df[df['sentiment_label'] == 'positive'])
        negative = len(df[df['sentiment_label'] == 'negative'])
        neutral = len(df[df['sentiment_label'] == 'neutral'])
        
        print(f"\nTotal texts analyzed: {total}")
        print(f"Positive: {positive} ({positive/total*100:.1f}%)")
        print(f"Negative: {negative} ({negative/total*100:.1f}%)")
        print(f"Neutral: {neutral} ({neutral/total*100:.1f}%)")
        
        print(f"\nAverage compound sentiment: {df['sentiment_compound'].mean():.3f}")
        print(f"Sentiment std deviation: {df['sentiment_compound'].std():.3f}")
        print(f"Most positive text score: {df['sentiment_compound'].max():.3f}")
        print(f"Most negative text score: {df['sentiment_compound'].min():.3f}")


def main():
    """Main execution"""
    print("="*60)
    print("BTC SENTIMENT ANALYZER")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Load data
    print("\n📂 Loading collected data...")
    data_dir = 'data'
    
    # Find most recent combined data file
    files = [f for f in os.listdir(data_dir) if f.startswith('combined_sentiment_data')]
    
    if not files:
        print("✗ No data files found!")
        print("Run sentiment_collector.py first to collect data")
        return
    
    latest_file = sorted(files)[-1]
    filepath = os.path.join(data_dir, latest_file)
    
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} texts")
    
    # Process sentiment
    df_analyzed = analyzer.process_dataframe(df)
    
    # Show summary
    analyzer.print_summary_stats(df_analyzed)
    
    # Create aggregations
    daily_agg = analyzer.aggregate_daily_sentiment(df_analyzed)
    source_features = analyzer.create_source_specific_features(df_analyzed)
    
    # Save results
    daily_path = analyzer.save_results(df_analyzed, daily_agg, source_features)
    
    print("\n" + "="*60)
    print("✅ SENTIMENT ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nNext step: Use {daily_path} in btc_predictor_enhanced.py")
    print("\nKey files created:")
    print("  - sentiment_daily_*.csv (USE THIS for modeling)")
    print("  - sentiment_by_source_*.csv (Reddit vs News comparison)")
    print("  - sentiment_detailed_*.csv (Full analysis)")


if __name__ == "__main__":
    main()