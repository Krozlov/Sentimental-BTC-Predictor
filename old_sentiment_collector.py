"""
BTC Sentiment Data Collector - Focused on CryptoPanic & RSS
Simplified and cleaned version with better error handling
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from bs4 import BeautifulSoup
import feedparser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class SentimentCollector:
    def __init__(self):
        """Initialize collector with session"""
        self.cryptopanic_key = None
        self.session = self._create_session()
        
    def _create_session(self):
        """Create session with retry logic"""
        session = requests.Session()
        retry = Retry(
            total=3, 
            backoff_factor=1, 
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def setup_cryptopanic(self, api_key=None):
        """Setup CryptoPanic API key (optional)"""
        self.cryptopanic_key = api_key
        if api_key:
            print("✓ CryptoPanic API key configured")
        else:
            print("✓ Using CryptoPanic free tier (no key)")
    
    def collect_cryptopanic_data(self, days_back=30):
        """Collect crypto news from CryptoPanic API with proper parameters"""
        print(f"\n📰 Collecting CryptoPanic data...")
        
        all_posts = []
        
        try:
            # Use API if key available, otherwise scrape
            if self.cryptopanic_key:
                print("  Using CryptoPanic API with proper parameters...")
                
                # CryptoPanic API documentation parameters
                # Base URL format: /api/developer/v2/posts/
                base_url = "https://cryptopanic.com/api/v2/posts/"
                
                # Proper parameter format according to docs
                params = {
                    'auth_token': self.cryptopanic_key,
                    'public': 'true',  # Required parameter
                    'currencies': 'BTC',  # Filter for Bitcoin only
                    'filter': 'hot',  # hot, rising, bullish, bearish, important, saved, lol
                    'kind': 'news',  # news or media
                    'regions': 'en',  # English content
                }
                
                print(f"    API Request: {base_url}")
                print(f"    Parameters: currencies=BTC, filter=hot, public=true")
                
                try:
                    response = self.session.get(
                        base_url, 
                        params=params, 
                        timeout=15
                    )
                    
                    # Debug response
                    print(f"    Response status: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        posts = data.get('results', [])
                        
                        if posts:
                            print(f"    ✓ API successful! Retrieved {len(posts)} posts")
                        else:
                            print(f"    ⚠ API returned 0 posts, trying fallback...")
                            posts = self._scrape_cryptopanic_public()
                    
                    elif response.status_code == 401:
                        print(f"    ✗ Authentication failed - invalid token")
                        print(f"    Get a valid token from: https://cryptopanic.com/developers/api/")
                        print(f"    Falling back to scraping...")
                        posts = self._scrape_cryptopanic_public()
                    
                    elif response.status_code == 403:
                        print(f"    ✗ Access forbidden - check token permissions")
                        print(f"    Falling back to scraping...")
                        posts = self._scrape_cryptopanic_public()
                    
                    elif response.status_code == 404:
                        print(f"    ✗ Endpoint not found - trying v1 API...")
                        # Fallback to v1 API
                        v1_url = "https://cryptopanic.com/api/v1/posts/"
                        v1_params = {
                            'auth_token': self.cryptopanic_key,
                            'currencies': 'BTC',
                            'public': 'true'
                        }
                        
                        v1_response = self.session.get(v1_url, params=v1_params, timeout=15)
                        if v1_response.status_code == 200:
                            data = v1_response.json()
                            posts = data.get('results', [])
                            print(f"    ✓ V1 API worked! Retrieved {len(posts)} posts")
                        else:
                            print(f"    ✗ V1 API also failed, falling back to scraping...")
                            posts = self._scrape_cryptopanic_public()
                    
                    else:
                        print(f"    ✗ Unexpected status code: {response.status_code}")
                        print(f"    Response: {response.text[:200]}")
                        print(f"    Falling back to scraping...")
                        posts = self._scrape_cryptopanic_public()
                        
                except requests.exceptions.RequestException as e:
                    print(f"    ✗ Request error: {str(e)[:100]}")
                    print(f"    Falling back to scraping...")
                    posts = self._scrape_cryptopanic_public()
                
            else:
                print("  No API token provided, using web scraping...")
                posts = self._scrape_cryptopanic_public()
            
            # Process posts
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for post in posts:
                try:
                    # Parse date
                    date_str = post.get('published_at', post.get('created_at', ''))
                    if date_str:
                        # Handle different date formats
                        if 'Z' in date_str or '+' in date_str or '-' in date_str[-6:]:
                            # ISO format with timezone
                            pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        else:
                            pub_date = datetime.now()
                    else:
                        pub_date = datetime.now()
                    
                    # Remove timezone for comparison
                    if pub_date.tzinfo:
                        pub_date = pub_date.replace(tzinfo=None)
                    
                    # Skip old posts
                    if pub_date < cutoff_date:
                        continue
                    
                    # Extract source name
                    source_info = post.get('source', {})
                    if isinstance(source_info, dict):
                        source_name = source_info.get('title', 'Unknown')
                    else:
                        source_name = 'Unknown'
                    
                    # Get sentiment votes if available
                    votes = post.get('votes', {})
                    if isinstance(votes, dict):
                        positive_votes = votes.get('positive', 0)
                        negative_votes = votes.get('negative', 0)
                        vote_score = positive_votes - negative_votes
                    else:
                        vote_score = 0
                    
                    all_posts.append({
                        'date': pub_date.strftime('%Y-%m-%d'),
                        'datetime': pub_date,
                        'source': f"CryptoPanic_{source_name}",
                        'title': post.get('title', ''),
                        'text': post.get('title', ''),  # CryptoPanic mainly has titles
                        'url': post.get('url', ''),
                        'vote_score': vote_score  # Extra: community sentiment
                    })
                    
                except Exception as e:
                    continue
            
            print(f"  ✓ Collected {len(all_posts)} CryptoPanic posts")
            
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            print("  Trying scraping as final fallback...")
            posts = self._scrape_cryptopanic_public()
            
            # Process scraped posts
            for post in posts[:100]:
                try:
                    all_posts.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'datetime': datetime.now(),
                        'source': 'CryptoPanic_Scraped',
                        'title': post.get('title', ''),
                        'text': post.get('title', ''),
                        'url': post.get('url', '')
                    })
                except:
                    continue
        
        return pd.DataFrame(all_posts)
    
    def _scrape_cryptopanic_public(self):
        """Scrape public CryptoPanic feed (fallback method)"""
        try:
            url = "https://cryptopanic.com/news/bitcoin/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            posts = []
            
            # Method 1: Try finding news cards (updated selectors)
            news_items = soup.find_all('div', class_='news-cell')
            if not news_items:
                news_items = soup.find_all('a', class_='nc-title')
            if not news_items:
                news_items = soup.find_all('div', class_='title-text')
            
            print(f"    Found {len(news_items)} items using scraping")
            
            for item in news_items[:100]:
                try:
                    # Try different ways to extract title and link
                    title = None
                    url_link = None
                    
                    # Method A: Direct title element
                    if item.name == 'a':
                        title = item.text.strip()
                        url_link = item.get('href', '')
                    else:
                        # Method B: Find anchor within div
                        title_elem = item.find('a')
                        if title_elem:
                            title = title_elem.text.strip()
                            url_link = title_elem.get('href', '')
                        else:
                            # Method C: Just get text
                            title = item.text.strip()
                            url_link = ''
                    
                    # Skip if no title
                    if not title or len(title) < 10:
                        continue
                    
                    # Make sure URL is absolute
                    if url_link and not url_link.startswith('http'):
                        url_link = 'https://cryptopanic.com' + url_link
                    
                    posts.append({
                        'title': title,
                        'url': url_link,
                        'published_at': datetime.now().isoformat(),
                        'source': {'title': 'CryptoPanic'}
                    })
                except Exception as e:
                    continue
            
            # Method 2: If still no posts, try getting all links with Bitcoin keywords
            if len(posts) < 10:
                print("    Trying alternative scraping method...")
                all_links = soup.find_all('a', href=True)
                for link in all_links:
                    try:
                        title = link.get('title', link.text.strip())
                        # Filter for Bitcoin-related content
                        if len(title) > 20 and any(word in title.lower() for word in ['bitcoin', 'btc', 'crypto']):
                            posts.append({
                                'title': title,
                                'url': link['href'] if link['href'].startswith('http') else 'https://cryptopanic.com' + link['href'],
                                'published_at': datetime.now().isoformat(),
                                'source': {'title': 'CryptoPanic'}
                            })
                            
                            if len(posts) >= 50:
                                break
                    except:
                        continue
            
            print(f"    Scraped {len(posts)} posts successfully")
            return posts[:100]  # Return max 100 posts
            
        except Exception as e:
            print(f"    Scraping failed: {e}")
            return []
    
    def collect_coindesk_rss(self, days_back=30):
        """Collect from CoinDesk and CoinTelegraph RSS feeds"""
        print(f"\n📡 Collecting RSS feeds...")
        
        all_posts = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        rss_urls = [
            ('https://www.coindesk.com/arc/outboundfeeds/rss/', 'CoinDesk'),
            ('https://cointelegraph.com/rss', 'CoinTelegraph'),
        ]
        
        for rss_url, source_name in rss_urls:
            try:
                print(f"  Fetching {source_name}...")
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:100]:
                    try:
                        # Parse date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = datetime.now()
                        
                        # Skip old posts
                        if pub_date < cutoff_date:
                            continue
                        
                        # Check if Bitcoin-related
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
                
                count = len([p for p in all_posts if source_name in p['source']])
                print(f"  ✓ Collected {count} {source_name} articles")
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ {source_name} failed: {e}")
        
        return pd.DataFrame(all_posts)
    
    def collect_bitcointalk_sentiment(self, pages=3):
        """Scrape BitcoinTalk forum"""
        print(f"\n💬 Collecting BitcoinTalk forum posts...")
        
        all_posts = []
        base_url = "https://bitcointalk.org/index.php?board=1.0"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            for page in range(pages):
                try:
                    url = f"{base_url};page={page}"
                    response = self.session.get(url, headers=headers, timeout=15)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find topic titles
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
                    
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"    Page {page} failed: {e}")
                    continue
            
            print(f"  ✓ Collected {len(all_posts)} BitcoinTalk topics")
            
        except Exception as e:
            print(f"  ✗ BitcoinTalk scraping failed: {e}")
        
        return pd.DataFrame(all_posts)
    
    def save_data(self, *dataframes, output_dir='data'):
        """Save collected data to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Filter valid dataframes
        valid_dfs = [df for df in dataframes if not df.empty]
        
        if not valid_dfs:
            print("\n✗ No data collected!")
            return None
        
        # Combine all data
        combined_df = pd.concat(valid_dfs, ignore_index=True)
        combined_df = combined_df.sort_values('datetime')
        
        # Save
        combined_path = f'{output_dir}/combined_sentiment_data_{timestamp}.csv'
        combined_df.to_csv(combined_path, index=False)
        
        print(f"\n💾 Combined data saved: {combined_path}")
        print(f"   Total records: {len(combined_df)}")
        
        return combined_path


def main():
    """Main execution"""
    print("="*60)
    print("BTC SENTIMENT DATA COLLECTOR")
    print("="*60)
    
    collector = SentimentCollector()
    
    # Get configuration
    days_back = int(input("\nHow many days of history? (default 30): ") or "30")
    
    print("\n" + "="*60)
    print("CRYPTOPANIC API SETUP (Optional)")
    print("="*60)
    print("CryptoPanic provides crypto-specific news sentiment.")
    print("\nTo use the API:")
    print("  1. Visit: https://cryptopanic.com/developers/api/")
    print("  2. Sign up for a free account")
    print("  3. Get your auth_token from the dashboard")
    print("  4. Use parameters: ?auth_token=TOKEN&public=true")
    print("\nWithout API key: Will use web scraping (slower, less data)")
    print("With API key: Fast, reliable, more posts")
    
    crypto_key = input("\nEnter CryptoPanic API token (or press Enter to skip): ").strip()
    
    if crypto_key:
        collector.setup_cryptopanic(crypto_key)
    else:
        collector.setup_cryptopanic(None)
    
    # Collect data
    print("\n" + "="*60)
    print("COLLECTING DATA FROM SOURCES")
    print("="*60)
    
    dataframes = []
    
    # Source 1: CryptoPanic (API or scraping)
    crypto_df = collector.collect_cryptopanic_data(days_back=days_back)
    if not crypto_df.empty:
        dataframes.append(crypto_df)
    
    # Source 2: RSS Feeds (CoinDesk, CoinTelegraph)
    rss_df = collector.collect_coindesk_rss(days_back=days_back)
    if not rss_df.empty:
        dataframes.append(rss_df)
    
    # Source 3: BitcoinTalk Forum
    forum_df = collector.collect_bitcointalk_sentiment(pages=3)
    if not forum_df.empty:
        dataframes.append(forum_df)
    
    # Save results
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
                    print(f"  • {source}: {count} items")
        
        print(f"\n📊 Total items collected: {sum(len(df) for df in dataframes)}")
        print(f"\n✅ Next step: Run sentiment_analyzer.py")
    else:
        print("\n⚠ No data collected. Try:")
        print("  1. Getting a CryptoPanic API token")
        print("  2. Checking your internet connection")
        print("  3. Reducing days_back to 14")


if __name__ == "__main__":
    main()