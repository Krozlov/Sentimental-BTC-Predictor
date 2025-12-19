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
        
        text = str(text)
        text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s!?.,]', '', text)  # Keep punctuation
        text = ' '.join(text.split())  # Remove extra whitespace
        
        return text
    
    def analyze_text(self, text):
        """Analyze sentiment using VADER"""
        cleaned = self.clean_text(text)
        
        if not cleaned:
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        
        return self.analyzer.polarity_scores(cleaned)
    
    def process_dataframe(self, df):
        """Process dataframe and add sentiment scores"""
        print("\n📊 Analyzing sentiment...")
        
        # Combine title and text
        if 'title' in df.columns:
            df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        else:
            df['full_text'] = df['text'].fillna('')
        
        # Analyze each text
        sentiments = []
        total = len(df)
        
        for idx, text in enumerate(df['full_text']):
            if idx % 100 == 0 and idx > 0:
                print(f"  Progress: {idx}/{total} texts...")
            
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
        """Aggregate sentiment scores by day"""
        print("\n📊 Aggregating daily sentiment scores...")
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date
        daily_agg = df.groupby('date').agg({
            'sentiment_compound': ['mean', 'std', 'min', 'max', 'count'],
            'sentiment_pos': 'mean',
            'sentiment_neg': 'mean',
            'sentiment_neu': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = [
            'date', 'sentiment_mean', 'sentiment_std', 
            'sentiment_min', 'sentiment_max', 'text_count',
            'sentiment_pos_mean', 'sentiment_neg_mean', 
            'sentiment_neu_mean'
        ]
        
        # Fill NaN std with 0 (for single data point days)
        daily_agg['sentiment_std'] = daily_agg['sentiment_std'].fillna(0)
        
        # Calculate sentiment momentum
        daily_agg = daily_agg.sort_values('date')
        daily_agg['sentiment_momentum'] = daily_agg['sentiment_mean'].diff().fillna(0)
        
        print(f"✓ Created daily aggregations for {len(daily_agg)} days")
        print(f"  Date range: {daily_agg['date'].min().date()} to {daily_agg['date'].max().date()}")
        
        return daily_agg
    
    def create_source_specific_features(self, df):
        """Create separate features for different sources"""
        print("\n🔀 Creating source-specific features...")
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Identify Reddit vs News vs Forum
        reddit_mask = df['source'].str.contains('Reddit', case=False, na=False)
        news_mask = df['source'].str.contains('News|RSS|CryptoPanic', case=False, na=False)
        forum_mask = df['source'].str.contains('Forum', case=False, na=False)
        
        # Aggregate by source type
        source_dfs = []
        
        if reddit_mask.any():
            reddit_daily = df[reddit_mask].groupby('date').agg({
                'sentiment_compound': 'mean'
            }).rename(columns={'sentiment_compound': 'reddit_sentiment'})
            source_dfs.append(reddit_daily)
        
        if news_mask.any():
            news_daily = df[news_mask].groupby('date').agg({
                'sentiment_compound': 'mean'
            }).rename(columns={'sentiment_compound': 'news_sentiment'})
            source_dfs.append(news_daily)
        
        if forum_mask.any():
            forum_daily = df[forum_mask].groupby('date').agg({
                'sentiment_compound': 'mean'
            }).rename(columns={'sentiment_compound': 'forum_sentiment'})
            source_dfs.append(forum_daily)
        
        # Combine all sources
        if source_dfs:
            source_features = pd.concat(source_dfs, axis=1)
            source_features = source_features.fillna(0)
            print(f"✓ Source-specific features created")
            return source_features.reset_index()
        else:
            print("⚠ No source-specific features created")
            return pd.DataFrame()
    
    def save_results(self, df_with_sentiment, daily_agg, source_features, output_dir='data'):
        """Save processed sentiment data"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed sentiment data
        detailed_path = f'{output_dir}/sentiment_detailed_{timestamp}.csv'
        df_with_sentiment.to_csv(detailed_path, index=False)
        print(f"\n💾 Detailed sentiment saved: {detailed_path}")
        
        # Save daily aggregated data
        daily_path = f'{output_dir}/sentiment_daily_{timestamp}.csv'
        daily_agg.to_csv(daily_path, index=False)
        print(f"💾 Daily sentiment saved: {daily_path}")
        
        # Save source-specific features
        if not source_features.empty:
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
        print(f"  • Positive: {positive} ({positive/total*100:.1f}%)")
        print(f"  • Negative: {negative} ({negative/total*100:.1f}%)")
        print(f"  • Neutral:  {neutral} ({neutral/total*100:.1f}%)")
        
        print(f"\nSentiment Statistics:")
        print(f"  • Average compound: {df['sentiment_compound'].mean():.3f}")
        print(f"  • Std deviation:    {df['sentiment_compound'].std():.3f}")
        print(f"  • Most positive:    {df['sentiment_compound'].max():.3f}")
        print(f"  • Most negative:    {df['sentiment_compound'].min():.3f}")
        
        # Show source breakdown
        if 'source' in df.columns:
            print(f"\nBy Source:")
            source_counts = df['source'].str.split('_').str[0].value_counts()
            for source, count in source_counts.items():
                print(f"  • {source}: {count} items")


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
    
    if not os.path.exists(data_dir):
        print("✗ Data directory not found!")
        print("Run sentiment_collector.py first to collect data")
        return
    
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
    
    # Verify required columns
    required_cols = ['date', 'text']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        return
    
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
    print(f"\nNext step: Use {daily_path} in btc_predictor.py")
    print("\nKey files created:")
    print("  • sentiment_daily_*.csv (USE THIS for modeling)")
    print("  • sentiment_by_source_*.csv (Source comparison)")
    print("  • sentiment_detailed_*.csv (Full analysis)")


if __name__ == "__main__":
    main()