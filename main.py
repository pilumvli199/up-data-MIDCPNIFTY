import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, timedelta
import json
import logging
import gzip
from typing import Dict, List, Optional, Tuple
from telegram import Bot
from telegram.error import TelegramError
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import mplfinance as mpf
from io import BytesIO
import pytz

# ======================== CONFIGURATION ========================
import os

UPSTOX_API_URL = "https://api.upstox.com/v2"
UPSTOX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# Trading params
ANALYSIS_INTERVAL = 5 * 60  # 5 minutes
CANDLES_COUNT = 200  # 200+ candles for better S/R
ATM_RANGE = 2  # ±2 strikes only (5 total)
STOP_LOSS_POINTS = 20  # 20 points for stop loss

# Market hours (IST)
MARKET_START = dt_time(9, 15)
MARKET_END = dt_time(15, 30)
IST = pytz.timezone('Asia/Kolkata')

INDICES = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================== CANDLESTICK PATTERN DETECTOR (VOLUME-WEIGHTED) ========================
class CandlestickPatterns:
    @staticmethod
    def is_hammer(row, prev_row=None, volume_avg=None) -> Tuple[bool, str, float]:
        """Hammer - Bullish reversal with volume confirmation"""
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, "", 0.0
        
        # Volume strength
        volume_strength = 1.0
        if volume_avg and row['volume'] > 0:
            volume_strength = min(row['volume'] / volume_avg, 2.0)
        
        # Hammer: small body, long lower wick, small upper wick
        if (lower_wick > body * 2 and 
            upper_wick < body * 0.3 and 
            body < total_range * 0.3):
            strength = volume_strength
            return True, "🔨 HAMMER", strength
        return False, "", 0.0
    
    @staticmethod
    def is_shooting_star(row, prev_row=None, volume_avg=None) -> Tuple[bool, str, float]:
        """Shooting Star - Bearish reversal with volume"""
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, "", 0.0
        
        volume_strength = 1.0
        if volume_avg and row['volume'] > 0:
            volume_strength = min(row['volume'] / volume_avg, 2.0)
        
        if (upper_wick > body * 2 and 
            lower_wick < body * 0.3 and 
            body < total_range * 0.3):
            strength = volume_strength
            return True, "⭐ SHOOTING STAR", strength
        return False, "", 0.0
    
    @staticmethod
    def is_doji(row, prev_row=None, volume_avg=None) -> Tuple[bool, str, float]:
        """Doji - Indecision"""
        body = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, "", 0.0
        
        volume_strength = 1.0
        if volume_avg and row['volume'] > 0:
            volume_strength = min(row['volume'] / volume_avg, 2.0)
        
        if body < total_range * 0.1:
            strength = volume_strength * 0.8  # Doji less reliable
            return True, "✖️ DOJI", strength
        return False, "", 0.0
    
    @staticmethod
    def is_engulfing(row, prev_row, volume_avg=None) -> Tuple[bool, str, float]:
        """Bullish/Bearish Engulfing"""
        if prev_row is None:
            return False, "", 0.0
        
        curr_body = abs(row['close'] - row['open'])
        prev_body = abs(prev_row['close'] - prev_row['open'])
        
        volume_strength = 1.0
        if volume_avg and row['volume'] > 0:
            volume_strength = min(row['volume'] / volume_avg, 2.0)
        
        # Bullish engulfing
        if (row['close'] > row['open'] and 
            prev_row['close'] < prev_row['open'] and
            row['open'] <= prev_row['close'] and
            row['close'] >= prev_row['open'] and
            curr_body > prev_body * 1.2):
            strength = volume_strength * 1.5  # Strong pattern
            return True, "🟢 BULLISH ENGULFING", strength
        
        # Bearish engulfing
        if (row['close'] < row['open'] and 
            prev_row['close'] > prev_row['open'] and
            row['open'] >= prev_row['close'] and
            row['close'] <= prev_row['open'] and
            curr_body > prev_body * 1.2):
            strength = volume_strength * 1.5
            return True, "🔴 BEARISH ENGULFING", strength
        
        return False, "", 0.0
    
    @staticmethod
    def is_morning_star(df, idx, volume_avg=None) -> Tuple[bool, str, float]:
        """Morning Star - Bullish reversal (3 candle)"""
        if idx < 2:
            return False, "", 0.0
        
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        
        first_red = first['close'] < first['open']
        second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
        third_green = third['close'] > third['open']
        third_closes_high = third['close'] > (first['open'] + first['close']) / 2
        
        if first_red and second_small and third_green and third_closes_high:
            volume_strength = 1.0
            if volume_avg and third['volume'] > 0:
                volume_strength = min(third['volume'] / volume_avg, 2.0)
            strength = volume_strength * 1.8  # Very strong pattern
            return True, "🌅 MORNING STAR", strength
        return False, "", 0.0
    
    @staticmethod
    def is_evening_star(df, idx, volume_avg=None) -> Tuple[bool, str, float]:
        """Evening Star - Bearish reversal (3 candle)"""
        if idx < 2:
            return False, "", 0.0
        
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        
        first_green = first['close'] > first['open']
        second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
        third_red = third['close'] < third['open']
        third_closes_low = third['close'] < (first['open'] + first['close']) / 2
        
        if first_green and second_small and third_red and third_closes_low:
            volume_strength = 1.0
            if volume_avg and third['volume'] > 0:
                volume_strength = min(third['volume'] / volume_avg, 2.0)
            strength = volume_strength * 1.8
            return True, "🌆 EVENING STAR", strength
        return False, "", 0.0
    
    @staticmethod
    def detect_all_patterns(df, futures_volume=0) -> List[Dict]:
        """Detect all patterns with volume weighting"""
        patterns = []
        
        # Calculate average volume
        volume_avg = df['volume'].mean() if 'volume' in df.columns else None
        
        # Use futures volume if available
        if futures_volume > 0 and volume_avg:
            volume_avg = (volume_avg + futures_volume) / 2
        
        for i in range(len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1] if i > 0 else None
            
            # Single candle patterns
            is_pat, name, strength = CandlestickPatterns.is_hammer(row, prev_row, volume_avg)
            if is_pat:
                patterns.append({
                    "index": i, 
                    "time": row.name, 
                    "pattern": name, 
                    "type": "bullish",
                    "strength": strength,
                    "price": row['close']
                })
                continue
            
            is_pat, name, strength = CandlestickPatterns.is_shooting_star(row, prev_row, volume_avg)
            if is_pat:
                patterns.append({
                    "index": i, 
                    "time": row.name, 
                    "pattern": name, 
                    "type": "bearish",
                    "strength": strength,
                    "price": row['close']
                })
                continue
            
            is_pat, name, strength = CandlestickPatterns.is_doji(row, prev_row, volume_avg)
            if is_pat:
                patterns.append({
                    "index": i, 
                    "time": row.name, 
                    "pattern": name, 
                    "type": "neutral",
                    "strength": strength,
                    "price": row['close']
                })
                continue
            
            # Two candle patterns
            if prev_row is not None:
                is_pat, name, strength = CandlestickPatterns.is_engulfing(row, prev_row, volume_avg)
                if is_pat:
                    pat_type = "bullish" if "BULLISH" in name else "bearish"
                    patterns.append({
                        "index": i, 
                        "time": row.name, 
                        "pattern": name, 
                        "type": pat_type,
                        "strength": strength,
                        "price": row['close']
                    })
                    continue
            
            # Three candle patterns
            is_pat, name, strength = CandlestickPatterns.is_morning_star(df, i, volume_avg)
            if is_pat:
                patterns.append({
                    "index": i, 
                    "time": row.name, 
                    "pattern": name, 
                    "type": "bullish",
                    "strength": strength,
                    "price": row['close']
                })
                continue
            
            is_pat, name, strength = CandlestickPatterns.is_evening_star(df, i, volume_avg)
            if is_pat:
                patterns.append({
                    "index": i, 
                    "time": row.name, 
                    "pattern": name, 
                    "type": "bearish",
                    "strength": strength,
                    "price": row['close']
                })
        
        return patterns

# ======================== SUPPORT & RESISTANCE DETECTOR ========================
class SupportResistance:
    @staticmethod
    def find_pivot_points(df, window=5) -> Tuple[List, List]:
        """Find pivot highs and lows - stronger with bigger window"""
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            if df.iloc[i]['high'] == df.iloc[i-window:i+window+1]['high'].max():
                highs.append((i, df.iloc[i]['high']))
            
            if df.iloc[i]['low'] == df.iloc[i-window:i+window+1]['low'].min():
                lows.append((i, df.iloc[i]['low']))
        
        return highs, lows
    
    @staticmethod
    def cluster_levels(levels, tolerance=0.003) -> List[float]:
        """Cluster nearby levels - tighter clustering"""
        if not levels:
            return []
        
        levels_sorted = sorted(levels)
        clusters = []
        current_cluster = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    @staticmethod
    def identify_levels(df, current_price) -> Dict:
        """Identify support and resistance from 200+ candles"""
        highs, lows = SupportResistance.find_pivot_points(df)
        
        high_prices = [h[1] for h in highs]
        low_prices = [l[1] for l in lows]
        
        resistance_levels = SupportResistance.cluster_levels(high_prices)
        support_levels = SupportResistance.cluster_levels(low_prices)
        
        # Filter and get top 3 each
        resistance_levels = [r for r in resistance_levels if r > current_price]
        support_levels = [s for s in support_levels if s < current_price]
        
        resistance_levels = sorted(resistance_levels)[:3]
        support_levels = sorted(support_levels, reverse=True)[:3]
        
        return {
            "resistance": resistance_levels,
            "support": support_levels
        }

# ======================== CONFLUENCE ANALYZER ========================
class ConfluenceAnalyzer:
    @staticmethod
    def calculate_score(pattern: Dict, sr_levels: Dict, strike_pcr: float, futures_volume: float, avg_volume: float) -> Dict:
        """
        Confluence Score (0-5):
        - Pattern at S/R: +2 points
        - PCR confirms direction: +2 points
        - High volume: +1 point
        """
        score = 0
        reasons = []
        
        pattern_price = pattern['price']
        pattern_type = pattern['type']
        
        # Check S/R proximity (within 0.5%)
        at_support = any(abs(pattern_price - s) / s < 0.005 for s in sr_levels['support'])
        at_resistance = any(abs(pattern_price - r) / r < 0.005 for r in sr_levels['resistance'])
        
        # S/R match (2 points)
        if pattern_type == 'bullish' and at_support:
            score += 2
            reasons.append("Pattern at Support")
        elif pattern_type == 'bearish' and at_resistance:
            score += 2
            reasons.append("Pattern at Resistance")
        
        # PCR confirmation (2 points)
        if pattern_type == 'bullish' and strike_pcr > 1.3:
            score += 2
            reasons.append(f"High PCR ({strike_pcr:.2f})")
        elif pattern_type == 'bearish' and strike_pcr < 0.7:
            score += 2
            reasons.append(f"Low PCR ({strike_pcr:.2f})")
        elif pattern_type == 'bullish' and strike_pcr > 1.0:
            score += 1
            reasons.append(f"Moderate PCR ({strike_pcr:.2f})")
        elif pattern_type == 'bearish' and strike_pcr < 1.0:
            score += 1
            reasons.append(f"Moderate PCR ({strike_pcr:.2f})")
        
        # Volume confirmation (1 point)
        volume_ratio = futures_volume / avg_volume if avg_volume > 0 else 1.0
        if volume_ratio > 1.5:
            score += 1
            reasons.append(f"High Volume ({volume_ratio:.1f}x)")
        
        # Strength
        if score >= 4:
            strength = "STRONG"
        elif score >= 2:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        return {
            "score": score,
            "strength": strength,
            "reasons": reasons
        }

# ======================== TRADING SIGNAL GENERATOR ========================
class TradingSignals:
    @staticmethod
    def generate_signal(pattern: Dict, confluence: Dict, current_price: float, strike: int, ce_ltp: float, pe_ltp: float) -> Optional[Dict]:
        """Generate trading signal with Entry/SL/Target"""
        
        # Only generate for MODERATE/STRONG confluence
        if confluence['score'] < 2:
            return None
        
        pattern_type = pattern['type']
        pattern_price = pattern['price']
        
        if pattern_type == 'bullish':
            # BUY PE
            entry = pattern_price
            stop_loss = entry - STOP_LOSS_POINTS
            target = entry + (STOP_LOSS_POINTS * 2.5)  # 2.5:1 reward
            
            option_type = "PE"
            option_price = pe_ltp
            action = "BUY"
            
        elif pattern_type == 'bearish':
            # BUY CE
            entry = pattern_price
            stop_loss = entry + STOP_LOSS_POINTS
            target = entry - (STOP_LOSS_POINTS * 2.5)
            
            option_type = "CE"
            option_price = ce_ltp
            action = "BUY"
        else:
            return None
        
        return {
            "action": action,
            "strike": strike,
            "option_type": option_type,
            "option_price": option_price,
            "entry": entry,
            "stop_loss": stop_loss,
            "target": target,
            "pattern": pattern['pattern'],
            "strength": confluence['strength'],
            "score": confluence['score'],
            "reasons": confluence['reasons']
        }

# ======================== UPSTOX API CLIENT (COMPLETE) ========================
class UpstoxClient:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.instruments_cache = None
        
    async def create_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    def get_instrument_key(self, symbol: str) -> str:
        mapping = {
            "NIFTY": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank",
            "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
            "FINNIFTY": "NSE_INDEX|Nifty Fin Service"
        }
        return mapping.get(symbol, f"NSE_EQ|{symbol}")
    
    def get_futures_key(self, symbol: str) -> str:
        """Get futures instrument key for volume data"""
        # Get nearest monthly expiry (last Thursday)
        today = datetime.now(IST).date()
        
        # Find last Thursday of current month
        last_day = (today.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        last_thursday = last_day - timedelta(days=(last_day.weekday() - 3) % 7)
        
        if last_thursday < today:
            # Move to next month
            next_month = (today.replace(day=28) + timedelta(days=4))
            last_day = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            last_thursday = last_day - timedelta(days=(last_day.weekday() - 3) % 7)
        
        expiry_str = last_thursday.strftime('%y%b').upper()  # 25JAN format
        
        futures_mapping = {
            "NIFTY": f"NSE_FO|NIFTY{expiry_str}FUT",
            "BANKNIFTY": f"NSE_FO|BANKNIFTY{expiry_str}FUT",
            "MIDCPNIFTY": f"NSE_FO|MIDCPNIFTY{expiry_str}FUT",
            "FINNIFTY": f"NSE_FO|FINNIFTY{expiry_str}FUT"
        }
        
        return futures_mapping.get(symbol, "")
    
    async def download_instruments(self):
        try:
            logger.info("📡 Downloading instruments...")
            
            async with self.session.get(UPSTOX_INSTRUMENTS_URL) as response:
                if response.status != 200:
                    logger.error(f"❌ Download failed: {response.status}")
                    return None
                
                content = await response.read()
                json_text = gzip.decompress(content).decode('utf-8')
                instruments = json.loads(json_text)
                
                logger.info(f"✅ Downloaded {len(instruments)} instruments")
                self.instruments_cache = instruments
                return instruments
                
        except Exception as e:
            logger.error(f"❌ Error downloading: {e}")
            return None
    
    async def get_available_expiries(self, symbol: str) -> List[str]:
        try:
            if not self.instruments_cache:
                await self.download_instruments()
            
            instruments = self.instruments_cache or []
            
            symbol_mapping = {
                "NIFTY": "NIFTY",
                "BANKNIFTY": "BANKNIFTY",
                "MIDCPNIFTY": "MIDCPNIFTY",
                "FINNIFTY": "FINNIFTY"
            }
            
            instrument_name = symbol_mapping.get(symbol, symbol)
            now = datetime.now(IST)
            expiries_set = set()
            
            for instrument in instruments:
                if (instrument.get('segment') == 'NSE_FO' and
                    instrument.get('instrument_type') in ['CE', 'PE'] and
                    instrument.get('name') == instrument_name):
                    
                    expiry_ms = instrument.get('expiry')
                    if expiry_ms:
                        try:
                            expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
                            if expiry_dt > now:
                                expiries_set.add(expiry_dt.strftime('%Y-%m-%d'))
                        except:
                            pass
            
            expiries = sorted(list(expiries_set))
            logger.info(f"   ✅ Found {len(expiries)} expiries")
            return expiries
                
        except Exception as e:
            logger.error(f"❌ Error finding expiries: {e}")
            return []
    
    async def get_full_quote(self, instrument_key: str) -> Dict:
        try:
            url = f"{UPSTOX_API_URL}/market-quote/quotes"
            params = {"instrument_key": instrument_key}
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("status") == "success" and data.get("data"):
                    for key, value in data["data"].items():
                        return {
                            "ltp": value.get("last_price", 0.0),
                            "volume": value.get("volume", 0),
                            "oi": value.get("oi", 0)
                        }
                
                return {"ltp": 0.0, "volume": 0, "oi": 0}
        except Exception as e:
            logger.error(f"❌ Quote error: {e}")
            return {"ltp": 0.0, "volume": 0, "oi": 0}
    
    async def get_ltp(self, instrument_key: str) -> float:
        quote = await self.get_full_quote(instrument_key)
        return quote["ltp"]
    
    async def get_futures_volume(self, symbol: str) -> float:
        """Get futures volume for volume confirmation"""
        try:
            futures_key = self.get_futures_key(symbol)
            if not futures_key:
                return 0.0
            
            quote = await self.get_full_quote(futures_key)
            volume = quote.get("volume", 0)
            
            logger.info(f"   📊 Futures Volume: {volume:,}")
            return float(volume)
            
        except Exception as e:
            logger.error(f"❌ Futures volume error: {e}")
            return 0.0
    
    async def get_historical_candles(self, instrument_key: str, count: int = 50) -> pd.DataFrame:
        """Get intraday 5min candles (recent data)"""
        try:
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            url = f"{UPSTOX_API_URL}/historical-candle/{instrument_key}/5minute/{to_date}"
            
            async with self.session.get(url) as response:
                data = await response.json()
                
                if data.get("status") == "success" and data.get("data", {}).get("candles"):
                    candles = data["data"]["candles"][:count]
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    
                    return df
                
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Intraday candles error: {e}")
            return pd.DataFrame()
    
    async def get_daily_candles(self, instrument_key: str, days: int = 30) -> pd.DataFrame:
        """Get daily candles for long-term S/R"""
        try:
            to_date = datetime.now(IST).strftime('%Y-%m-%d')
            url = f"{UPSTOX_API_URL}/historical-candle/{instrument_key}/day/{to_date}"
            
            async with self.session.get(url) as response:
                data = await response.json()
                
                if data.get("status") == "success" and data.get("data", {}).get("candles"):
                    candles = data["data"]["candles"][:days]
                    
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    
                    return df
                
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Daily candles error: {e}")
            return pd.DataFrame()
    
    async def get_combined_candles(self, instrument_key: str) -> pd.DataFrame:
        """Combine daily + intraday for 200+ candles"""
        try:
            logger.info("   📈 Fetching 200+ candles (daily + intraday)...")
            
            # Get daily candles (30 days)
            daily_df = await self.get_daily_candles(instrument_key, days=30)
            
            # Get intraday candles (today)
            intraday_df = await self.get_historical_candles(instrument_key, count=100)
            
            if daily_df.empty and intraday_df.empty:
                return pd.DataFrame()
            
            # Resample daily to 5min for consistency (use close prices)
            if not daily_df.empty:
                daily_5min = daily_df.resample('5T').ffill()
                combined = pd.concat([daily_5min, intraday_df]).sort_index()
                combined = combined[~combined.index.duplicated(keep='last')]
            else:
                combined = intraday_df
            
            # Take last 200
            combined = combined.tail(200)
            
            logger.info(f"   ✅ Combined: {len(combined)} candles")
            return combined
            
        except Exception as e:
            logger.error(f"❌ Combined candles error: {e}")
            return pd.DataFrame()
    
    async def get_option_contracts(self, symbol: str, expiry: str) -> List[Dict]:
        try:
            instrument_key = self.get_instrument_key(symbol)
            url = f"{UPSTOX_API_URL}/option/contract"
            
            params = {
                "instrument_key": instrument_key,
                "expiry_date": expiry
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("status") == "success":
                    contracts = data.get("data", [])
                    logger.info(f"   ✅ Fetched {len(contracts)} contracts")
                    return contracts
                
                return []
                    
        except Exception as e:
            logger.error(f"❌ Contracts error: {e}")
            return []

# ======================== OPTION CHAIN ANALYZER (COMPLETE) ========================
class OptionAnalyzer:
    def __init__(self, client: UpstoxClient):
        self.client = client
    
    def get_strike_interval(self, symbol: str) -> int:
        intervals = {
            "NIFTY": 50,
            "BANKNIFTY": 100,
            "MIDCPNIFTY": 25,
            "FINNIFTY": 50
        }
        return intervals.get(symbol, 100)
    
    async def filter_atm_strikes(self, contracts: List[Dict], current_price: float, symbol: str) -> Dict:
        """±2 strikes only (5 total)"""
        interval = self.get_strike_interval(symbol)
        atm = round(current_price / interval) * interval
        min_strike = atm - (ATM_RANGE * interval)
        max_strike = atm + (ATM_RANGE * interval)
        
        logger.info(f"   🎯 ATM: {atm}, Range: {min_strike} to {max_strike}")
        
        ce_contracts = {}
        pe_contracts = {}
        
        for contract in contracts:
            strike = contract.get("strike_price")
            if min_strike <= strike <= max_strike:
                instrument_key = contract.get("instrument_key")
                option_type = contract.get("instrument_type")
                
                contract_data = {
                    "strike": strike,
                    "instrument_key": instrument_key,
                    "trading_symbol": contract.get("trading_symbol"),
                    "ltp": 0,
                    "oi": 0,
                    "volume": 0
                }
                
                if option_type == "CE":
                    ce_contracts[strike] = contract_data
                elif option_type == "PE":
                    pe_contracts[strike] = contract_data
        
        return {
            "ce": ce_contracts,
            "pe": pe_contracts,
            "strikes": sorted(set(list(ce_contracts.keys()) + list(pe_contracts.keys())))
        }
    
    async def fetch_option_prices(self, contracts_data: Dict):
        for strike, contract in contracts_data["ce"].items():
            quote = await self.client.get_full_quote(contract["instrument_key"])
            contract["ltp"] = quote["ltp"]
            contract["oi"] = quote["oi"]
            contract["volume"] = quote["volume"]
            await asyncio.sleep(0.05)
        
        for strike, contract in contracts_data["pe"].items():
            quote = await self.client.get_full_quote(contract["instrument_key"])
            contract["ltp"] = quote["ltp"]
            contract["oi"] = quote["oi"]
            contract["volume"] = quote["volume"]
            await asyncio.sleep(0.05)
    
    def find_nearest_strike(self, price: float, strikes: List[int]) -> int:
        """Find nearest strike to pattern formation price"""
        return min(strikes, key=lambda x: abs(x - price))
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            logger.info(f"📊 Analyzing {symbol}...")
            
            instrument_key = self.client.get_instrument_key(symbol)
            current_price = await self.client.get_ltp(instrument_key)
            
            if current_price == 0:
                logger.warning(f"⚠️ No price for {symbol}")
                return None
            
            logger.info(f"   💰 Spot: ₹{current_price:,.2f}")
            
            # Get futures volume
            futures_volume = await self.client.get_futures_volume(symbol)
            
            # Get 200+ candles (daily + intraday)
            candles = await self.client.get_combined_candles(instrument_key)
            
            if candles.empty:
                logger.warning(f"⚠️ No candle data")
                return None
            
            logger.info(f"   📈 Using {len(candles)} candles")
            
            # Calculate average volume
            avg_volume = candles['volume'].mean()
            
            # Detect patterns with volume weighting
            patterns = CandlestickPatterns.detect_all_patterns(candles, futures_volume)
            logger.info(f"   🎯 Found {len(patterns)} patterns")
            
            # Identify S/R from 200+ candles
            sr_levels = SupportResistance.identify_levels(candles, current_price)
            logger.info(f"   📊 S/R identified")
            
            # Get option contracts
            expiries = await self.client.get_available_expiries(symbol)
            
            if not expiries:
                logger.warning(f"⚠️ No expiries")
                return None
            
            contracts = []
            expiry = None
            
            for exp_date in expiries[:3]:
                contracts = await self.client.get_option_contracts(symbol, exp_date)
                if contracts:
                    expiry = exp_date
                    break
            
            if not contracts or not expiry:
                logger.warning(f"⚠️ No contracts")
                return None
            
            contracts_data = await self.filter_atm_strikes(contracts, current_price, symbol)
            
            if not contracts_data["strikes"]:
                logger.warning(f"⚠️ No strikes")
                return None
            
            await self.fetch_option_prices(contracts_data)
            
            # Calculate PCR for each strike
            for strike in contracts_data["strikes"]:
                ce = contracts_data["ce"].get(strike)
                pe = contracts_data["pe"].get(strike)
                
                if ce and pe:
                    strike_pcr = pe["oi"] / ce["oi"] if ce["oi"] > 0 else 0
                    ce["pcr"] = strike_pcr
                    pe["pcr"] = strike_pcr
            
            # Link patterns to strikes and generate signals
            high_probability_setups = []
            
            for pattern in patterns[-10:]:  # Recent patterns only
                nearest_strike = self.find_nearest_strike(pattern['price'], contracts_data["strikes"])
                
                ce = contracts_data["ce"].get(nearest_strike, {})
                pe = contracts_data["pe"].get(nearest_strike, {})
                strike_pcr = ce.get("pcr", 0)
                
                # Calculate confluence
                confluence = ConfluenceAnalyzer.calculate_score(
                    pattern, sr_levels, strike_pcr, futures_volume, avg_volume
                )
                
                # Generate trading signal
                signal = TradingSignals.generate_signal(
                    pattern, confluence, current_price, nearest_strike,
                    ce.get("ltp", 0), pe.get("ltp", 0)
                )
                
                if signal:
                    high_probability_setups.append({
                        "pattern": pattern,
                        "strike": nearest_strike,
                        "pcr": strike_pcr,
                        "confluence": confluence,
                        "signal": signal
                    })
            
            # Sort by confluence score
            high_probability_setups.sort(key=lambda x: x['confluence']['score'], reverse=True)
            
            # Build final data
            ce_data = []
            pe_data = []
            
            for s in contracts_data["strikes"]:
                ce = contracts_data["ce"].get(s, {"strike": s, "ltp": 0, "oi": 0, "volume": 0, "pcr": 0})
                pe = contracts_data["pe"].get(s, {"strike": s, "ltp": 0, "oi": 0, "volume": 0, "pcr": 0})
                ce_data.append(ce)
                pe_data.append(pe)
            
            logger.info(f"   🎯 Generated {len(high_probability_setups)} trading signals")
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "expiry": expiry,
                "candles": candles,
                "patterns": patterns,
                "sr_levels": sr_levels,
                "strikes": contracts_data["strikes"],
                "ce_data": ce_data,
                "pe_data": pe_data,
                "futures_volume": futures_volume,
                "avg_volume": avg_volume,
                "high_probability_setups": high_probability_setups
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            return None

# ======================== ENHANCED CHART GENERATOR (24x16, 4 SECTIONS) ========================
class ChartGenerator:
    @staticmethod
    def create_combined_chart(analysis: Dict) -> BytesIO:
        """24x16 chart with 4 sections"""
        symbol = analysis["symbol"]
        candles = analysis["candles"].tail(50)  # Show last 50 on chart
        current_price = analysis["current_price"]
        patterns = analysis.get("patterns", [])
        sr_levels = analysis.get("sr_levels", {"support": [], "resistance": []})
        high_prob_setups = analysis.get("high_probability_setups", [])
        
        # ✅ 24x16 inches, DPI 220
        fig = plt.figure(figsize=(24, 16), facecolor='white', dpi=220)
        gs = GridSpec(4, 1, height_ratios=[3, 0.8, 0.8, 1.2], hspace=0.35)
        
        # ========== 1. CANDLESTICK CHART ==========
        ax1 = fig.add_subplot(gs[0])
        
        mc = mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='inherit',
            wick={'up': '#26a69a', 'down': '#ef5350'},
            volume='in',
            alpha=0.9
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle='--',
            gridcolor='#e0e0e0',
            facecolor='white',
            figcolor='white',
            y_on_right=False
        )
        
        mpf.plot(
            candles,
            type='candle',
            style=s,
            ax=ax1,
            volume=False,
            show_nontrading=False
        )
        
        # Draw S/R
        for support in sr_levels['support']:
            ax1.axhline(y=support, color='green', linestyle='--', linewidth=2.5, alpha=0.7)
            ax1.text(0.01, support, f'  🟢 Support ₹{support:.0f}', 
                    transform=ax1.get_yaxis_transform(), 
                    color='white', fontsize=11, fontweight='bold', va='center',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='green', alpha=0.9))
        
        for resistance in sr_levels['resistance']:
            ax1.axhline(y=resistance, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
            ax1.text(0.01, resistance, f'  🔴 Resistance ₹{resistance:.0f}', 
                    transform=ax1.get_yaxis_transform(), 
                    color='white', fontsize=11, fontweight='bold', va='center',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='red', alpha=0.9))
        
        # Mark patterns
        for pattern in patterns[-8:]:
            idx = pattern['index']
            if idx >= len(candles):
                continue
            
            candle_time = candles.index[min(idx, len(candles)-1)]
            candle_high = candles.iloc[min(idx, len(candles)-1)]['high']
            
            color = 'green' if pattern['type'] == 'bullish' else ('red' if pattern['type'] == 'bearish' else 'blue')
            marker = '▲' if pattern['type'] == 'bullish' else ('▼' if pattern['type'] == 'bearish' else '●')
            
            ax1.annotate(
                f"{marker}", 
                xy=(candle_time, candle_high),
                xytext=(0, 25),
                textcoords='offset points',
                fontsize=16,
                fontweight='bold',
                color=color,
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor=color, linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color=color, linewidth=2)
            )
        
        ax1.set_title(f"{symbol} - 5min Chart (Last 50 of 200+ candles) | Spot: ₹{current_price:,.2f} | Expiry: {analysis['expiry']}", 
                     fontsize=18, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linewidth=0.8)
        
        # ========== 2. HIGH PROBABILITY SETUPS ==========
        ax_setups = fig.add_subplot(gs[1])
        ax_setups.axis('off')
        
        setup_text = "🎯 HIGH PROBABILITY TRADING SETUPS (Top 3):\n"
        
        if high_prob_setups:
            for i, setup in enumerate(high_prob_setups[:3], 1):
                signal = setup['signal']
                conf = setup['confluence']
                pattern = setup['pattern']
                
                time_str = pattern['time'].strftime('%H:%M')
                
                setup_text += f"\n#{i}. {signal['action']} {signal['strike']} {signal['option_type']} @ ₹{signal['option_price']:.2f}\n"
                setup_text += f"    Entry: ₹{signal['entry']:.0f} | SL: ₹{signal['stop_loss']:.0f} | Target: ₹{signal['target']:.0f}\n"
                setup_text += f"    Pattern: {signal['pattern']} ({time_str}) | Strength: {signal['strength']} (Score: {signal['score']}/5)\n"
                setup_text += f"    PCR: {setup['pcr']:.2f} | Reasons: {', '.join(conf['reasons'])}\n"
        else:
            setup_text += "  No high-probability setups at the moment.\n"
        
        ax_setups.text(0.02, 0.95, setup_text, transform=ax_setups.transAxes,
                      fontsize=11, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='orange', linewidth=2))
        
        # ========== 3. PATTERN ANALYSIS ==========
        ax_patterns = fig.add_subplot(gs[2])
        ax_patterns.axis('off')
        
        pattern_text = "📍 RECENT PATTERNS & S/R LEVELS:\n"
        
        recent_patterns = patterns[-5:]
        if recent_patterns:
            for p in recent_patterns:
                time_str = p['time'].strftime('%H:%M')
                pattern_text += f"  • {time_str} - {p['pattern']} @ ₹{p['price']:.0f} (Strength: {p['strength']:.1f})\n"
        
        pattern_text += f"\n📊 KEY LEVELS:\n"
        for support in sr_levels['support']:
            pattern_text += f"  🟢 Support: ₹{support:.0f}\n"
        for resistance in sr_levels['resistance']:
            pattern_text += f"  🔴 Resistance: ₹{resistance:.0f}\n"
        
        pattern_text += f"\n📈 Volume: Futures={analysis['futures_volume']:,.0f} | Avg={analysis['avg_volume']:,.0f}"
        
        ax_patterns.text(0.02, 0.95, pattern_text, transform=ax_patterns.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # ========== 4. OPTION CHAIN TABLE ==========
        ax_table = fig.add_subplot(gs[3])
        ax_table.axis('tight')
        ax_table.axis('off')
        
        interval = 50 if symbol == "NIFTY" else (100 if symbol == "BANKNIFTY" else 50)
        atm_strike = round(current_price / interval) * interval
        
        table_data = [["Strike", "PE OI", "CE OI", "PCR", "CE LTP", "PE LTP", "Signal"]]
        
        for i, strike in enumerate(analysis["strikes"]):
            ce = analysis["ce_data"][i]
            pe = analysis["pe_data"][i]
            
            pcr = pe.get("pcr", 0)
            
            if pcr > 2.0:
                signal = "🟢🟢 STRONG BUY"
            elif pcr > 1.5:
                signal = "🟢 Bullish"
            elif pcr > 1.1:
                signal = "⚪ Slight Bull"
            elif pcr >= 0.9:
                signal = "⚪ Neutral"
            elif pcr >= 0.6:
                signal = "🔴 Bearish"
            else:
                signal = "🔴🔴 STRONG SELL"
            
            row = [
                f"₹{strike:,.0f}{' *ATM*' if strike == atm_strike else ''}",
                f"{pe['oi']:,}",
                f"{ce['oi']:,}",
                f"{pcr:.2f}",
                f"₹{ce['ltp']:.2f}",
                f"₹{pe['ltp']:.2f}",
                signal
            ]
            table_data.append(row)
        
        total_ce_oi = sum(d["oi"] for d in analysis["ce_data"])
        total_pe_oi = sum(d["oi"] for d in analysis["pe_data"])
        overall_pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        if overall_pcr > 1.5:
            market_signal = "🟢 BULLISH MARKET"
        elif overall_pcr > 1.1:
            market_signal = "🟢 Slight Bullish"
        elif overall_pcr >= 0.9:
            market_signal = "⚪ NEUTRAL"
        else:
            market_signal = "🔴 BEARISH MARKET"
        
        table_data.append([
            "OVERALL PCR",
            f"{total_pe_oi:,.0f}",
            f"{total_ce_oi:,.0f}",
            f"{overall_pcr:.2f}",
            "",
            "",
            market_signal
        ])
        
        table = ax_table.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.14, 0.13, 0.13, 0.10, 0.12, 0.12, 0.16]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3)
        
        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#2c3e50')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
        
        # Style summary row
        summary_row = len(table_data) - 1
        for i in range(7):
            table[(summary_row, i)].set_facecolor('#f39c12')
            table[(summary_row, i)].set_text_props(weight='bold', fontsize=12)
        
        # Highlight ATM
        for i, strike in enumerate(analysis["strikes"], 1):
            if strike == atm_strike:
                for j in range(7):
                    table[(i, j)].set_facecolor('#d4edda')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=220, facecolor='white', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

# ======================== TELEGRAM ALERTER ========================
class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
    
    async def send_chart(self, chart_buffer: BytesIO, symbol: str, analysis: Dict):
        try:
            high_prob_setups = analysis.get('high_probability_setups', [])
            
            caption = f"""📊 {symbol} COMPLETE ANALYSIS

💰 Spot: ₹{analysis['current_price']:,.2f}
📅 Expiry: {analysis['expiry']}
📈 Candles: {len(analysis['candles'])} (200+ combined)

"""
            
            if high_prob_setups:
                caption += "🎯 TOP TRADING SETUP:\n"
                top = high_prob_setups[0]
                signal = top['signal']
                
                caption += f"{signal['action']} {signal['strike']} {signal['option_type']} @ ₹{signal['option_price']:.2f}\n"
                caption += f"Entry: ₹{signal['entry']:.0f} | SL: ₹{signal['stop_loss']:.0f} | Target: ₹{signal['target']:.0f}\n"
                caption += f"Strength: {signal['strength']} ({signal['score']}/5)\n"
                caption += f"Pattern: {signal['pattern']}\n"
            
            total_ce_oi = sum(d["oi"] for d in analysis["ce_data"])
            total_pe_oi = sum(d["oi"] for d in analysis["pe_data"])
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            caption += f"\n📊 Overall PCR: {pcr:.3f}\n"
            caption += f"📈 Futures Vol: {analysis['futures_volume']:,.0f}\n"
            caption += f"\n⏰ {datetime.now(IST).strftime('%d-%b %H:%M IST')}"
            
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=chart_buffer,
                caption=caption
            )
            
            logger.info(f"✅ Alert sent for {symbol}")
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
        except Exception as e:
            logger.error(f"❌ Send error: {e}")

# ======================== MAIN BOT ========================
class UpstoxOptionsBot:
    def __init__(self):
        self.client = UpstoxClient(UPSTOX_ACCESS_TOKEN)
        self.analyzer = OptionAnalyzer(self.client)
        self.chart_gen = ChartGenerator()
        self.alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    def is_market_open(self) -> bool:
        now = datetime.now(IST).time()
        today = datetime.now(IST).date()
        
        if today.weekday() >= 5:
            return False
            
        return MARKET_START <= now <= MARKET_END
    
    async def process_symbols(self):
        now_time = datetime.now(IST)
        
        if not self.is_market_open():
            logger.info(f"⏸️ Market closed | {now_time.strftime('%H:%M:%S IST')}")
            return
        
        logger.info("\n" + "="*70)
        logger.info(f"🔍 ANALYSIS CYCLE - {now_time.strftime('%H:%M:%S IST')}")
        logger.info("="*70)
        
        for symbol in INDICES:
            try:
                analysis = await self.analyzer.analyze_symbol(symbol)
                
                if analysis:
                    chart = self.chart_gen.create_combined_chart(analysis)
                    await self.alerter.send_chart(chart, symbol, analysis)
                    logger.info(f"✅ {symbol} complete\n")
                else:
                    logger.warning(f"⚠️ {symbol} failed\n")
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error {symbol}: {e}\n")
        
        logger.info("="*70)
        logger.info("✅ CYCLE COMPLETE")
        logger.info("="*70 + "\n")
    
    async def run(self):
        current_time = datetime.now(IST)
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        
        print("\n" + "="*70)
        print("🚀 UPSTOX OPTIONS BOT - COMPLETE v4.0", flush=True)
        print("="*70)
        print(f"📅 {current_time.strftime('%d-%b-%Y %A')}", flush=True)
        print(f"🕐 {current_time.strftime('%H:%M:%S IST')}", flush=True)
        print(f"📊 Market: {market_status}", flush=True)
        print(f"⏱️  Interval: {ANALYSIS_INTERVAL//60} minutes", flush=True)
        print("", flush=True)
        print("✅ 24x16 Chart (DPI 220)", flush=True)
        print("✅ ±2 Strikes (5 total)", flush=True)
        print("✅ 200+ Candles (Historical + Intraday)", flush=True)
        print("✅ Futures Volume Confirmation", flush=True)
        print("✅ Pattern-Strike Linking", flush=True)
        print("✅ Confluence Score (0-5)", flush=True)
        print("✅ Trading Signals (Entry/SL/Target)", flush=True)
        print("✅ High Probability Setups", flush=True)
        print("="*70 + "\n", flush=True)
        
        await self.client.create_session()
        
        try:
            while True:
                try:
                    await self.process_symbols()
                    
                    next_run = datetime.now(IST) + timedelta(seconds=ANALYSIS_INTERVAL)
                    logger.info(f"⏰ Next: {next_run.strftime('%H:%M:%S IST')}\n")
                    
                    await asyncio.sleep(ANALYSIS_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"❌ Loop error: {e}")
                    await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped")
        
        finally:
            await self.client.close_session()
            logger.info("👋 Session closed")

# ======================== ENTRY POINT ========================
if __name__ == "__main__":
    bot = UpstoxOptionsBot()
    asyncio.run(bot.run())
