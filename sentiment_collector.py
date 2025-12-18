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
        """Collect crypto news from CryptoPanic API with pagination for historical data"""
        print(f"\n📰 Collecting CryptoPanic data (targeting {days_back} days)...")
        
        all_posts = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        try:
            # Use API if key available, otherwise scrape
            if self.cryptopanic_key:
                print("  Using CryptoPanic API with pagination...")
                
                base_url = "https://cryptopanic.com/api/v2/posts/"
                next_page = None
                page_count = 0
                max_pages = 20  # Limit to prevent infinite loops
                
                while page_count < max_pages:
                    page_count += 1
                    
                    # Build parameters
                    params = {
                        'auth_token': self.cryptopanic_key,
                        'public': 'true',
                        'currencies': 'BTC',
                        'filter': 'hot',
                        'kind': 'news',
                    }
                    
                    # Use cursor for pagination if available
                    if next_page:
                        # CryptoPanic uses cursor-based pagination
                        # Extract the cursor from the next URL
                        if 'cursor=' in next_page:
                            cursor = next_page.split('cursor=')[1].split('&')[0]
                            params['cursor'] = cursor
                    
                    try:
                        response = self.session.get(base_url, params=params, timeout=15)
                        
                        if response.status_code == 200:
                            data = response.json()
                            posts = data.get('results', [])
                            
                            if not posts:
                                print(f"    Page {page_count}: No more posts")
                                break
                            
                            # Process posts and check dates
                            oldest_date = None
                            posts_added = 0
                            
                            for post in posts:
                                try:
                                    date_str = post.get('published_at', post.get('created_at', ''))
                                    if date_str:
                                        if 'Z' in date_str or '+' in date_str or '-' in date_str[-6:]:
                                            pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                        else:
                                            pub_date = datetime.now()
                                    else:
                                        pub_date = datetime.now()
                                    
                                    if pub_date.tzinfo:
                                        pub_date = pub_date.replace(tzinfo=None)
                                    
                                    # Track oldest date
                                    if oldest_date is None or pub_date < oldest_date:
                                        oldest_date = pub_date
                                    
                                    # Skip if too old
                                    if pub_date < cutoff_date:
                                        continue
                                    
                                    # Extract info
                                    source_info = post.get('source', {})
                                    source_name = source_info.get('title', 'Unknown') if isinstance(source_info, dict) else 'Unknown'
                                    
                                    votes = post.get('votes', {})
                                    if isinstance(votes, dict):
                                        vote_score = votes.get('positive', 0) - votes.get('negative', 0)
                                    else:
                                        vote_score = 0
                                    
                                    all_posts.append({
                                        'date': pub_date.strftime('%Y-%m-%d'),
                                        'datetime': pub_date,
                                        'source': f"CryptoPanic_{source_name}",
                                        'title': post.get('title', ''),
                                        'text': post.get('title', ''),
                                        'url': post.get('url', ''),
                                        'vote_score': vote_score
                                    })
                                    posts_added += 1
                                    
                                except Exception as e:
                                    continue
                            
                            # Show progress
                            days_collected = (datetime.now() - oldest_date).days if oldest_date else 0
                            print(f"    Page {page_count}: +{posts_added} posts (oldest: {days_collected} days ago)")
                            
                            # Check if we've gone back far enough
                            if oldest_date and oldest_date < cutoff_date:
                                print(f"    ✓ Reached target date range!")
                                break
                            
                            # Get next page URL
                            next_page = data.get('next')
                            if not next_page:
                                print(f"    No more pages available")
                                break
                            
                        elif response.status_code == 404:
                            # Try V1 API
                            print(f"    V2 API failed, trying V1...")
                            v1_url = "https://cryptopanic.com/api/v1/posts/"
                            v1_response = self.session.get(v1_url, params=params, timeout=15)
                            if v1_response.status_code == 200:
                                data = v1_response.json()
                                posts = data.get('results', [])
                                # Process V1 posts (same logic as above)
                                for post in posts[:100]:
                                    try:
                                        date_str = post.get('published_at', '')
                                        if date_str:
                                            pub_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                            if pub_date.tzinfo:
                                                pub_date = pub_date.replace(tzinfo=None)
                                            if pub_date >= cutoff_date:
                                                all_posts.append({
                                                    'date': pub_date.strftime('%Y-%m-%d'),
                                                    'datetime': pub_date,
                                                    'source': 'CryptoPanic_API',
                                                    'title': post.get('title', ''),
                                                    'text': post.get('title', ''),
                                                    'url': post.get('url', '')
                                                })
                                    except:
                                        continue
                            break
                        else:
                            print(f"    API error: {response.status_code}")
                            break
                            
                    except Exception as e:
                        print(f"    Page {page_count} error: {str(e)[:50]}")
                        break
                    
                    # Rate limiting
                    time.sleep(1)
                
                print(f"  ✓ API collected {len(all_posts)} posts across {page_count} pages")
                
            else:
                print("  No API token, using web scraping...")
                posts = self._scrape_cryptopanic_public()
                
                # Process scraped posts
                for post in posts:
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
            
            # If API gave us very little data, supplement with scraping
            if self.cryptopanic_key and len(all_posts) < 50:
                print(f"  ⚠ Only got {len(all_posts)} posts from API, supplementing with scraping...")
                scraped = self._scrape_cryptopanic_public()
                for post in scraped:
                    all_posts.append({
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'datetime': datetime.now(),
                        'source': 'CryptoPanic_Scraped',
                        'title': post.get('title', ''),
                        'text': post.get('title', ''),
                        'url': post.get('url', '')
                    })
            
            # Show date range collected
            if all_posts:
                df_temp = pd.DataFrame(all_posts)
                date_range = df_temp['date'].nunique()
                print(f"  ✓ Total: {len(all_posts)} posts covering {date_range} unique dates")
            else:
                print(f"  ✗ No posts collected")
            
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
        
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
        """Collect from CoinDesk and CoinTelegraph RSS feeds (usually limited to ~7-14 days)"""
        print(f"\n📡 Collecting RSS feeds (note: RSS typically limited to last 7-14 days)...")
        
        all_posts = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Multiple RSS sources for better coverage
        rss_sources = [
            ('https://www.coindesk.com/arc/outboundfeeds/rss/', 'CoinDesk'),
            ('https://cointelegraph.com/rss', 'CoinTelegraph'),
            ('https://cryptopotato.com/feed/', 'CryptoPotato'),
            ('https://bitcoinmagazine.com/.rss/full/', 'BitcoinMagazine'),
            ('https://www.newsbtc.com/feed/', 'NewsBTC'),
        ]
        
        for rss_url, source_name in rss_sources:
            try:
                print(f"  Fetching {source_name}...")
                feed = feedparser.parse(rss_url)
                
                items_collected = 0
                
                for entry in feed.entries[:200]:  # Increased from 100 to 200
                    try:
                        # Parse date
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            pub_date = datetime(*entry.updated_parsed[:6])
                        else:
                            pub_date = datetime.now()
                        
                        # Skip old posts
                        if pub_date < cutoff_date:
                            continue
                        
                        # Check if Bitcoin-related
                        title_text = entry.title.lower()
                        summary_text = entry.get('summary', entry.get('description', '')).lower()
                        
                        if any(word in title_text or word in summary_text 
                              for word in ['bitcoin', 'btc', 'crypto']):
                            all_posts.append({
                                'date': pub_date.strftime('%Y-%m-%d'),
                                'datetime': pub_date,
                                'source': f'RSS_{source_name}',
                                'title': entry.title,
                                'text': entry.get('summary', entry.get('description', entry.title)),
                                'url': entry.link
                            })
                            items_collected += 1
                    except Exception as e:
                        continue
                
                print(f"    ✓ Collected {items_collected} {source_name} articles")
                time.sleep(1)  # Be respectful
                
            except Exception as e:
                print(f"    ✗ {source_name} failed: {str(e)[:50]}")
        
        # Show date range
        if all_posts:
            df_temp = pd.DataFrame(all_posts)
            date_range = df_temp['date'].nunique()
            oldest = df_temp['datetime'].min()
            days_covered = (datetime.now() - oldest).days
            print(f"\n  ✓ RSS Total: {len(all_posts)} articles covering {date_range} dates ({days_covered} days back)")
            print(f"  ℹ️  Note: RSS feeds typically only provide last 7-14 days of data")
        
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
    print("\n" + "="*60)
    print("⚠️  IMPORTANT: DATA LIMITATIONS")
    print("="*60)
    print("Most free news sources only provide recent data:")
    print("  • CryptoPanic API: Usually 2-7 days of posts per page")
    print("  • RSS Feeds: Typically 7-14 days maximum")
    print("  • BitcoinTalk: Current posts only")
    print("\nTo get 30 days of data:")
    print("  1. Use CryptoPanic API with pagination (best option)")
    print("  2. Run collector daily to build historical dataset")
    print("  3. Consider paid APIs for full historical access")
    print("="*60)
    
    days_back = int(input("\nHow many days to target? (default 30): ") or "30")
    
    if days_back > 14:
        print(f"\n⚠️  Warning: Requested {days_back} days, but may only get 7-14 days")
        print("This is normal - free APIs have limited history")
    
    print("\n" + "="*60)
    print("CRYPTOPANIC API SETUP (Recommended)")
    print("="*60)
    print("CryptoPanic provides crypto-specific news sentiment.")
    print("\nTo use the API:")
    print("  1. Visit: https://cryptopanic.com/developers/api/")
    print("  2. Sign up for a free account")
    print("  3. Get your auth_token from the dashboard")
    print("  4. Paste it below")
    print("\nWithout API: Will use web scraping (very limited)")
    print("With API: Better data, pagination for more days")
    
    crypto_key = input("\nEnter CryptoPanic API token (or press Enter to skip): ").strip()
    
    if crypto_key:
        collector.setup_cryptopanic(crypto_key)
    else:
        collector.setup_cryptopanic(None)
        print("\n⚠️  Running without API - expect minimal data!")
    
    # Collect data
    print("\n" + "="*60)
    print("COLLECTING DATA FROM SOURCES")
    print("="*60)
    
    dataframes = []
    
    # Source 1: CryptoPanic (API or scraping)
    crypto_df = collector.collect_cryptopanic_data(days_back=days_back)
    if not crypto_df.empty:
        dataframes.append(crypto_df)
    
    # Source 2: RSS Feeds (CoinDesk, CoinTelegraph, etc.)
    rss_df = collector.collect_coindesk_rss(days_back=days_back)
    if not rss_df.empty:
        dataframes.append(rss_df)
    
    # Source 3: BitcoinTalk Forum (current posts)
    forum_df = collector.collect_bitcointalk_sentiment(pages=3)
    if not forum_df.empty:
        dataframes.append(forum_df)
    
    # Save results
    if dataframes:
        saved_path = collector.save_data(*dataframes)
        
        print("\n" + "="*60)
        print("✅ DATA COLLECTION COMPLETE!")
        print("="*60)
        
        # Calculate actual date coverage
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        unique_dates = combined_df['date'].nunique()
        oldest_date = combined_df['date'].min()
        newest_date = combined_df['date'].max()
        days_span = (newest_date - oldest_date).days + 1
        
        print(f"\n📊 Collection Summary:")
        print(f"  • Total items: {len(combined_df)}")
        print(f"  • Unique dates: {unique_dates}")
        print(f"  • Date range: {oldest_date.date()} to {newest_date.date()}")
        print(f"  • Days covered: {days_span} days")
        
        print(f"\n📈 Data by source:")
        for df in dataframes:
            if not df.empty:
                sources = df['source'].str.split('_').str[0].unique()
                for source in sources:
                    count = len(df[df['source'].str.contains(source)])
                    source_dates = df[df['source'].str.contains(source)]['date'].nunique() if 'date' in df.columns else 'N/A'
                    print(f"  • {source}: {count} items across {source_dates} dates")
        
        if days_span < days_back:
            print(f"\n⚠️  Note: Got {days_span} days instead of {days_back} days")
            print("This is expected - free APIs have limited historical data")
            print("Run this script daily to build up historical dataset!")
        
        print(f"\n✅ Next step: Run sentiment_analyzer.py")
    else:
        print("\n⚠️  No data collected. Try:")
        print("  1. Getting a CryptoPanic API token (essential!)")
        print("  2. Checking your internet connection")
        print("  3. Running during business hours (more active news)")


if __name__ == "__main__":
    main()