#!/usr/bin/env python3
"""
Fetch all posts from @tolk_tolk Telegram channel.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import tz

try:
    from telethon import TelegramClient
    from telethon.errors import (
        UsernameNotOccupiedError,
        ChannelPrivateError,
        FloodWaitError,
    )
except ImportError:
    print("Error: telethon not installed. Run: pip install telethon")
    sys.exit(1)

# API credentials
API_ID = 25604239
API_HASH = "be0d223a25ba43ac03f64d658b577d2c"
SESSION_FILE = "data/telegram_session"
CHANNEL_USERNAME = "tolk_tolk"

async def fetch_all_posts(client: TelegramClient, username: str) -> list[dict]:
    """Fetch all posts from a Telegram channel."""
    posts = []
    
    try:
        print(f"📡 Connecting to @{username}...")
        entity = await client.get_entity(username)
        print(f"✅ Connected. Fetching posts...")
        
        total_fetched = 0
        async for msg in client.iter_messages(entity, limit=None):
            if not (msg.message or msg.text):
                continue
            
            post = {
                'message_id': msg.id,
                'date_utc': msg.date.isoformat(),
                'text': msg.message or msg.text or '',
                'views': msg.views or 0,
                'forwards': msg.forwards or 0,
                'replies': getattr(msg.replies, 'replies', None) if msg.replies else None,
                'has_media': msg.media is not None,
            }
            
            posts.append(post)
            total_fetched += 1
            
            if total_fetched % 100 == 0:
                print(f"   Fetched {total_fetched} posts...")
    
    except UsernameNotOccupiedError:
        print(f"❌ Channel @{username} not found")
    except ChannelPrivateError:
        print(f"🔒 Channel @{username} is private")
    except FloodWaitError as e:
        print(f"⏳ Rate limit: wait {e.seconds} seconds (~{e.seconds/3600:.1f} hours)")
        if posts:
            print(f"   Saving {len(posts)} posts before waiting...")
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return posts

async def main():
    output_file = Path("data/raw/tolk_posts.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Fetching posts from @{CHANNEL_USERNAME}\n")
    
    client = TelegramClient(SESSION_FILE, API_ID, API_HASH)
    
    try:
        await client.start()
        print("✅ Connected to Telegram\n")
        
        posts = await fetch_all_posts(client, CHANNEL_USERNAME)
        
        if posts:
            import pandas as pd
            
            df = pd.DataFrame(posts)
            
            # Sort by date (newest first)
            df['date_utc'] = pd.to_datetime(df['date_utc'])
            df = df.sort_values('date_utc', ascending=False)
            
            # Save
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            print(f"\n✅ Saved {len(df):,} posts to {output_file}")
            print(f"   Date range: {df['date_utc'].min()} to {df['date_utc'].max()}")
            
            # Show sample
            print("\n📄 Sample posts:")
            for i, row in df.head(3).iterrows():
                text_preview = row['text'][:100].replace('\n', ' ')
                print(f"   [{row['date_utc'].strftime('%Y-%m-%d')}] {text_preview}...")
        else:
            print("⚠️  No posts found")
    
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())


