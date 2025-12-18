"""
BTC Sentiment Data Collector - NO REDDIT API NEEDED
Uses CryptoPanic API + NewsAPI + Web Scraping
Run this FIRST to gather sentiment data
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class SentimentCollector:
    def __init__(self):
        """Initialize collectors"""
        self.news_api_key = "5016d1659dfb4eef9be53e1f0fbc260c";
        self.cryptopanic_key = "7aaa2e50ce410ce34d33d3dffac742c1c68c0f4c";
        self.session = self._create_session()
        
    def _create_session(self):
        """Create session with retry logic"""
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def setup_news_api(self, api_key):
        """Setup News API key"""
        self.news_api_key = api_key
        print("✓ News API key configured")
    
    def setup_cryptopanic(self, api_key=None):
        """
        Setup CryptoPanic API
        Free tier: No API key needed for public endpoint!
        Pro tier (optional): Get key from https://cryptopanic.com/developers/api/
        """
        self.cryptopanic_key = api_key
        if api_key:
            print("✓ CryptoPanic API key configured (Pro tier)")
        else:
            print("✓ CryptoPanic configured (Free tier - no key needed)")
    
    def collect_cryptopanic_data(self, days_back=30, posts_per_page=50):
        """
        Collect crypto news from CryptoPanic
        This is specifically designed for crypto sentiment!
        """
        print(f"\n📰 Collecting CryptoPanic data (crypto-specific news)...")
        
        all_posts = []
        
        # CryptoPanic free API endpoint
        base_url = "https://cryptopanic.com/api/v1/posts/"
        
        params = {
            'auth_token': self.cryptopanic_key if self.cryptopanic_key else 'free',
            'currencies': 'BTC',
            'kind': 'news',  # or 'media' for social media posts
            'filter': 'hot'  # hot, rising, bullish, bearish
        }
        
        try:
            # Try to get data even without API key (some endpoints work)
            # Use RSS feed as fallback
            rss_url = "https://cryptopanic.com/api/v1/posts/?auth_token=public&currencies=BTC&kind=news&public=true"
            
            print("  Fetching crypto news...")
            
            # Method 1: Try API endpoint
            try:
                if self.cryptopanic_key:
                    response = self.session.get(base_url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    posts = data.get('results', [])
                else:
                    # Method 2: Scrape public feed
                    posts = self._scrape_cryptopanic_public()
                
                for post in posts[:100]:  # Limit to 100 posts
                    try:
                        pub_date = datetime.fromisoformat(post.get('published_at', '').replace('Z', '+00:00'))
                        
                        if (datetime.now(pub_date.tzinfo) - pub_date).days > days_back:
                            continue
                        
                        all_posts.append({
                            'date': pub_date.strftime('%Y-%m-%d'),
                            'datetime': pub_date,
                            'source': f"CryptoPanic_{post.get('source', {}).get('title', 'Unknown')}",
                            'title': post.get('title', ''),
                            'text': post.get('title', ''),  # CryptoPanic only has titles
                            'url': post.get('url', ''),
                            'votes': post.get('votes', {}).get('positive', 0) - post.get('votes', {}).get('negative', 0)
                        })
                    except Exception as e:
                        continue
                
                print(f"    ✓ Collected {len(all_posts)} CryptoPanic posts")
                
            except Exception as e:
                print(f"    ⚠ CryptoPanic API error: {e}")
                print("    Trying alternative method...")
                
        except Exception as e:
            print(f"    ✗ CryptoPanic collection failed: {e}")
        
        return pd.DataFrame(all_posts)
    
    def _scrape_cryptopanic_public(self):
        """Scrape public CryptoPanic feed (no API needed)"""
        try:
            url = "https://cryptopanic.com/news/bitcoin/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            posts = []
            news_items = soup.find_all('div', class_='news-row')[:50]  # Get first 50
            
            for item in news_items:
                try:
                    title_elem = item.find('a', class_='title')
                    if title_elem:
                        posts.append({
                            'title': title_elem.text.strip(),
                            'url': title_elem.get('href', ''),
                            'published_at': datetime.now().isoformat(),
                            'source': {'title': 'CryptoPanic'},
                            'votes': {'positive': 0, 'negative': 0}
                        })
                except:
                    continue
            
            return posts
            
        except Exception as e:
            print(f"    Scraping failed: {e}")
            return []
    
    def collect_coindesk_rss(self, days_back=30):
        """
        Collect from CoinDesk RSS feed (no API needed!)
        CoinDesk is a major crypto news source
        """
        print(f"\n📡 Collecting CoinDesk RSS feed...")
        
        import feedparser
        
        all_posts = []
        
        rss_urls = [
            'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'https://cointelegraph.com/rss',
        ]
        
        for rss_url in rss_urls:
            try:
                feed = feedparser.parse(rss_url)
                source_name = "CoinDesk" if "coindesk" in rss_url else "CoinTelegraph"
                
                for entry in feed.entries[:100]:
                    try:
                        if hasattr(entry, 'published_parsed'):
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()
                        
                        if (datetime.now() - pub_date).days > days_back:
                            continue
                        
                        # Only include Bitcoin-related articles
                        title_text = entry.title.lower()
                        summary_text = entry.get('summary', '').lower()
                        
                        if any(word in title_text or word in summary_text 
                              for word in ['bitcoin', 'btc', 'crypto']):
                            all_posts.append({
                                'date': pub_date.strftime('%Y-%m-%d'),
                                'datetime': pub_date,
                                'source': f'RSS_{source_name}',
                                'title': entry.title,
                                'text': entry.get('summary', entry.title),
                                'url': entry.link
                            })
                    except Exception as e:
                        continue
                
                print(f"    ✓ Collected {len([p for p in all_posts if source_name in p['source']])} {source_name} articles")
                time.sleep(1)
                
            except Exception as e:
                print(f"    ✗ {rss_url} failed: {e}")
        
        return pd.DataFrame(all_posts)
    
    def collect_news_data(self, query='bitcoin OR btc OR cryptocurrency', 
                         days_back=30, page_size=100):
        """
        Collect news articles using NewsAPI
        """
        if not self.news_api_key:
            print("⚠ News API not configured. Skipping...")
            return pd.DataFrame()
        
        print(f"\n📰 Collecting NewsAPI articles...")
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': page_size,
            'apiKey': self.news_api_key
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = data.get('articles', [])
            
            news_list = []
            for article in articles:
                pub_date = datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                
                news_list.append({
                    'date': pub_date.strftime('%Y-%m-%d'),
                    'datetime': pub_date,
                    'source': f"NewsAPI_{article['source']['name']}",
                    'title': article['title'],
                    'text': article.get('description', '') or article['title'],
                    'url': article['url']
                })
            
            df = pd.DataFrame(news_list)
            print(f"✓ Total NewsAPI articles collected: {len(df)}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"✗ NewsAPI error: {e}")
            return pd.DataFrame()
    
    def collect_bitcointalk_sentiment(self, pages=5):
        """
        Scrape BitcoinTalk forum (largest Bitcoin forum)
        No API needed!
        """
        print(f"\n💬 Collecting BitcoinTalk forum posts...")
        
        all_posts = []
        base_url = "https://bitcointalk.org/index.php?board=1.0"  # Bitcoin Discussion
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            for page in range(pages):
                try:
                    url = f"{base_url};page={page}"
                    response = self.session.get(url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    topics = soup.find_all('td', class_='subject')
                    
                    for topic in topics:
                        try:
                            title_elem = topic.find('a')
                            if title_elem:
                                all_posts.append({
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'datetime': datetime.now(),
                                    'source': 'Forum_BitcoinTalk',
                                    'title': title_elem.text.strip(),
                                    'text': title_elem.text.strip(),
                                    'url': 'https://bitcointalk.org' + title_elem.get('href', '')
                                })
                        except:
                            continue
                    
                    time.sleep(2)  # Be respectful to the server
                    
                except Exception as e:
                    print(f"    Page {page} failed: {e}")
                    continue
            
            print(f"    ✓ Collected {len(all_posts)} BitcoinTalk topics")
            
        except Exception as e:
            print(f"    ✗ BitcoinTalk scraping failed: {e}")
        
        return pd.DataFrame(all_posts)
    
    def save_data(self, *dataframes, output_dir='data'):
        """Save collected data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Combine all dataframes
        valid_dfs = [df for df in dataframes if not df.empty]
        
        if not valid_dfs:
            print("\n✗ No data collected!")
            return None
        
        combined_df = pd.concat(valid_dfs, ignore_index=True)
        combined_df = combined_df.sort_values('datetime')
        
        combined_path = f'{output_dir}/combined_sentiment_data_{timestamp}.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"\n💾 Combined data saved: {combined_path}")
        print(f"   Total records: {len(combined_df)}")
        
        return combined_path


def main():
    """
    Main execution function
    NO REDDIT API NEEDED!
    """
    print("="*60)
    print("BTC SENTIMENT DATA COLLECTOR (No Reddit)")
    print("="*60)
    
    collector = SentimentCollector()
    
    # Configuration
    days_back = int(input("\nHow many days of history? (recommend 30): ") or "30")
    
    print("\n" + "="*60)
    print("DATA SOURCE CONFIGURATION")
    print("="*60)
    
    # OPTION 1: NewsAPI (optional but recommended)
    print("\n[1] NewsAPI (optional)")
    print("Get free API key from: https://newsapi.org/register")
    news_key = input("Enter News API KEY (or press Enter to skip): ").strip()
    if news_key:
        collector.setup_news_api(news_key)
    
    # OPTION 2: CryptoPanic (optional, has free tier)
    print("\n[2] CryptoPanic (optional - works without key)")
    print("Free tier works without API key!")
    print("For more data, get key from: https://cryptopanic.com/developers/api/")
    crypto_key = input("Enter CryptoPanic API KEY (or press Enter for free tier): ").strip()
    collector.setup_cryptopanic(crypto_key if crypto_key else None)
    
    # Collect data from all sources
    print("\n" + "="*60)
    print("COLLECTING DATA FROM MULTIPLE SOURCES")
    print("="*60)
    
    dataframes = []
    
    # Source 1: CryptoPanic
    crypto_df = collector.collect_cryptopanic_data(days_back=days_back)
    if not crypto_df.empty:
        dataframes.append(crypto_df)
    
    # Source 2: RSS Feeds (CoinDesk, CoinTelegraph)
    print("\nInstalling feedparser for RSS...")
    try:
        import feedparser
    except ImportError:
        os.system('pip install feedparser')
        import feedparser
    
    rss_df = collector.collect_coindesk_rss(days_back=days_back)
    if not rss_df.empty:
        dataframes.append(rss_df)
    
    # Source 3: NewsAPI (if configured)
    if collector.news_api_key:
        news_df = collector.collect_news_data(days_back=days_back)
        if not news_df.empty:
            dataframes.append(news_df)
    
    # Source 4: BitcoinTalk Forum
    print("\nInstalling BeautifulSoup for web scraping...")
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        os.system('pip install beautifulsoup4')
    
    forum_df = collector.collect_bitcointalk_sentiment(pages=3)
    if not forum_df.empty:
        dataframes.append(forum_df)
    
    # Save all data
    if dataframes:
        saved_path = collector.save_data(*dataframes)
        
        print("\n" + "="*60)
        print("✅ DATA COLLECTION COMPLETE!")
        print("="*60)
        print(f"\nData sources used:")
        for df in dataframes:
            if not df.empty:
                sources = df['source'].str.split('_').str[0].unique()
                for source in sources:
                    count = len(df[df['source'].str.contains(source)])
                    print(f"  - {source}: {count} items")
        
        print(f"\n📊 Total items collected: {sum(len(df) for df in dataframes)}")
        print(f"\nNext step: Run sentiment_analyzer.py")
    else:
        print("\n⚠ No data collected. Try:")
        print("  1. Using NewsAPI (free, instant)")
        print("  2. Checking your internet connection")
        print("  3. Reducing days_back to 14")


if __name__ == "__main__":
    main()