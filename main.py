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
from collections import deque

# ======================== CONFIGURATION ========================
import os

UPSTOX_API_URL = "https://api.upstox.com/v2"
UPSTOX_API_V3_URL = "https://api.upstox.com/v3"
UPSTOX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# Trading params
ANALYSIS_INTERVAL = 5 * 60  # 5 minutes
CANDLES_COUNT = 200  # Increased for better S/R
ATM_RANGE = 2  # ±2 strikes only

# Market hours (IST)
MARKET_START = dt_time(9, 15)
MARKET_END = dt_time(15, 30)
IST = pytz.timezone('Asia/Kolkata')

# ✅ ALL MAJOR INDICES
INDICES = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]

# ✅ EXPIRY DAY MAPPING (As per NSE Sept 2025 changes - ALL on TUESDAY)
EXPIRY_DAYS = {
    "NIFTY": 1,        # Tuesday (0=Monday, 1=Tuesday, ..., 6=Sunday)
    "BANKNIFTY": 1,    # Tuesday
    "FINNIFTY": 1,     # Tuesday
    "MIDCPNIFTY": 1,   # Tuesday
}

# ✅ STRIKE INTERVALS
STRIKE_INTERVALS = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
    "MIDCPNIFTY": 25,
}

# ✅ LOT SIZES (for reference)
LOT_SIZES = {
    "NIFTY": 25,
    "BANKNIFTY": 15,
    "FINNIFTY": 40,
    "MIDCPNIFTY": 75,
}

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================== STRONGEST CANDLESTICK PATTERNS ========================
class CandlestickPatterns:
    """Top 6 strongest patterns only"""
    
    @staticmethod
    def is_hammer(row, prev_row=None) -> Tuple[bool, str, str]:
        """Hammer - Strong Bullish Reversal"""
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, "", ""
        
        if (lower_wick > body * 2 and 
            upper_wick < body * 0.3 and 
            body < total_range * 0.3):
            return True, "🔨 HAMMER", "BULLISH"
        return False, "", ""
    
    @staticmethod
    def is_shooting_star(row, prev_row=None) -> Tuple[bool, str, str]:
        """Shooting Star - Strong Bearish Reversal"""
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, "", ""
        
        if (upper_wick > body * 2 and 
            lower_wick < body * 0.3 and 
            body < total_range * 0.3):
            return True, "⭐ SHOOTING STAR", "BEARISH"
        return False, "", ""
    
    @staticmethod
    def is_engulfing(row, prev_row) -> Tuple[bool, str, str]:
        """Bullish/Bearish Engulfing - Very Strong"""
        if prev_row is None:
            return False, "", ""
        
        curr_body = abs(row['close'] - row['open'])
        prev_body = abs(prev_row['close'] - prev_row['open'])
        
        # Bullish engulfing
        if (row['close'] > row['open'] and 
            prev_row['close'] < prev_row['open'] and
            row['open'] <= prev_row['close'] and
            row['close'] >= prev_row['open'] and
            curr_body > prev_body * 1.2):
            return True, "🟢 BULLISH ENGULFING", "BULLISH"
        
        # Bearish engulfing
        if (row['close'] < row['open'] and 
            prev_row['close'] > prev_row['open'] and
            row['open'] >= prev_row['close'] and
            row['close'] <= prev_row['open'] and
            curr_body > prev_body * 1.2):
            return True, "🔴 BEARISH ENGULFING", "BEARISH"
        
        return False, "", ""
    
    @staticmethod
    def is_morning_star(df, idx) -> Tuple[bool, str, str]:
        """Morning Star - Strong Bullish Reversal"""
        if idx < 2:
            return False, "", ""
        
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        
        first_red = first['close'] < first['open']
        second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
        third_green = third['close'] > third['open']
        third_closes_high = third['close'] > (first['open'] + first['close']) / 2
        
        if first_red and second_small and third_green and third_closes_high:
            return True, "🌅 MORNING STAR", "BULLISH"
        return False, "", ""
    
    @staticmethod
    def is_evening_star(df, idx) -> Tuple[bool, str, str]:
        """Evening Star - Strong Bearish Reversal"""
        if idx < 2:
            return False, "", ""
        
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        
        first_green = first['close'] > first['open']
        second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
        third_red = third['close'] < third['open']
        third_closes_low = third['close'] < (first['open'] + first['close']) / 2
        
        if first_green and second_small and third_red and third_closes_low:
            return True, "🌆 EVENING STAR", "BEARISH"
        return False, "", ""
    
    @staticmethod
    def is_doji(row, prev_row=None) -> Tuple[bool, str, str]:
        """Doji - Reversal Warning"""
        body = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, "", ""
        
        if body < total_range * 0.1:
            return True, "✖️ DOJI", "NEUTRAL"
        return False, "", ""
    
    @staticmethod
    def detect_patterns_with_volume(df, volume_data) -> List[Dict]:
        """Detect patterns with volume confirmation"""
        patterns = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1] if i > 0 else None
            
            # Get volume for this candle
            candle_volume = volume_data.get(row.name, 0) if volume_data else 0
            avg_volume = np.mean(list(volume_data.values())) if volume_data else 0
            high_volume = candle_volume > avg_volume * 1.2 if avg_volume > 0 else False
            
            # Check patterns (strongest first)
            is_pat, name, bias = CandlestickPatterns.is_engulfing(row, prev_row)
            if is_pat:
                patterns.append({
                    "index": i,
                    "time": row.name,
                    "pattern": name,
                    "type": bias.lower(),
                    "price": row['close'],
                    "high_volume": high_volume,
                    "volume": candle_volume
                })
                continue
            
            is_pat, name, bias = CandlestickPatterns.is_morning_star(df, i)
            if is_pat:
                patterns.append({
                    "index": i,
                    "time": row.name,
                    "pattern": name,
                    "type": bias.lower(),
                    "price": row['close'],
                    "high_volume": high_volume,
                    "volume": candle_volume
                })
                continue
            
            is_pat, name, bias = CandlestickPatterns.is_evening_star(df, i)
            if is_pat:
                patterns.append({
                    "index": i,
                    "time": row.name,
                    "pattern": name,
                    "type": bias.lower(),
                    "price": row['close'],
                    "high_volume": high_volume,
                    "volume": candle_volume
                })
                continue
            
            is_pat, name, bias = CandlestickPatterns.is_hammer(row, prev_row)
            if is_pat:
                patterns.append({
                    "index": i,
                    "time": row.name,
                    "pattern": name,
                    "type": bias.lower(),
                    "price": row['close'],
                    "high_volume": high_volume,
                    "volume": candle_volume
                })
                continue
            
            is_pat, name, bias = CandlestickPatterns.is_shooting_star(row, prev_row)
            if is_pat:
                patterns.append({
                    "index": i,
                    "time": row.name,
                    "pattern": name,
                    "type": bias.lower(),
                    "price": row['close'],
                    "high_volume": high_volume,
                    "volume": candle_volume
                })
                continue
            
            is_pat, name, bias = CandlestickPatterns.is_doji(row, prev_row)
            if is_pat:
                patterns.append({
                    "index": i,
                    "time": row.name,
                    "pattern": name,
                    "type": bias.lower(),
                    "price": row['close'],
                    "high_volume": high_volume,
                    "volume": candle_volume
                })
        
        return patterns

# ======================== SUPPORT & RESISTANCE ========================
class SupportResistance:
    @staticmethod
    def find_pivot_points(df, window=5) -> Tuple[List, List]:
        """Find pivot highs and lows"""
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
        """Cluster nearby levels"""
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
        """Identify support and resistance levels"""
        highs, lows = SupportResistance.find_pivot_points(df)
        
        high_prices = [h[1] for h in highs]
        low_prices = [l[1] for l in lows]
        
        resistance_levels = SupportResistance.cluster_levels(high_prices)
        support_levels = SupportResistance.cluster_levels(low_prices)
        
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
    """Analyze Pattern + S/R + PCR confluence"""
    
    @staticmethod
    def find_nearest_strike(price, interval):
        """Find nearest strike to price"""
        return round(price / interval) * interval
    
    @staticmethod
    def analyze_pattern_confluence(pattern, sr_levels, strike_data, atm_strike, symbol):
        """Analyze if pattern has confluence with S/R and PCR"""
        pattern_price = pattern['price']
        pattern_type = pattern['type']
        pattern_name = pattern['pattern']
        high_volume = pattern.get('high_volume', False)
        
        # Find nearest strike
        interval = STRIKE_INTERVALS.get(symbol, 50)
        nearest_strike = ConfluenceAnalyzer.find_nearest_strike(pattern_price, interval)
        
        # Get strike PCR
        strike_info = strike_data.get(nearest_strike)
        if not strike_info:
            return None
        
        ce_oi = strike_info.get('ce_oi', 0)
        pe_oi = strike_info.get('pe_oi', 0)
        pcr = pe_oi / ce_oi if ce_oi > 0 else 0
        
        # Check S/R confluence
        tolerance = interval * 0.6  # 60% of strike interval
        near_support = any(abs(pattern_price - s) < tolerance for s in sr_levels['support'])
        near_resistance = any(abs(pattern_price - r) < tolerance for r in sr_levels['resistance'])
        
        # Calculate confluence score
        score = 0
        reasons = []
        
        # Pattern type check
        if pattern_type == "bullish":
            if pcr > 1.3:
                score += 2
                reasons.append(f"High PCR {pcr:.2f} confirms bullish")
            if near_support:
                score += 2
                reasons.append("At support level")
        
        elif pattern_type == "bearish":
            if pcr < 0.8:
                score += 2
                reasons.append(f"Low PCR {pcr:.2f} confirms bearish")
            if near_resistance:
                score += 2
                reasons.append("At resistance level")
        
        # Volume confirmation
        if high_volume:
            score += 1
            reasons.append("High volume confirmation")
        
        # Determine signal strength
        if score >= 4:
            strength = "STRONG"
        elif score >= 2:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        return {
            "pattern": pattern_name,
            "type": pattern_type,
            "price": pattern_price,
            "time": pattern['time'],
            "nearest_strike": nearest_strike,
            "strike_pcr": pcr,
            "ce_oi": ce_oi,
            "pe_oi": pe_oi,
            "near_support": near_support,
            "near_resistance": near_resistance,
            "high_volume": high_volume,
            "score": score,
            "strength": strength,
            "reasons": reasons
        }
    
    @staticmethod
    def generate_trade_signal(confluence, atm_strike, symbol):
        """Generate actionable trade signal"""
        if confluence['strength'] == "WEAK":
            return None
        
        pattern_type = confluence['type']
        nearest_strike = confluence['nearest_strike']
        entry_price = confluence['price']
        
        # Dynamic stop loss based on index volatility
        interval = STRIKE_INTERVALS.get(symbol, 50)
        stop_distance = interval * 0.4  # 40% of strike interval
        target_distance = interval * 1.0  # 100% of strike interval
        
        # Calculate stop loss and target
        if pattern_type == "bullish":
            stop_loss = entry_price - stop_distance
            target = entry_price + target_distance
            
            # Find tradeable strike
            trade_strike = nearest_strike
            option_type = "PE"
            action = "BUY"
            
        else:  # bearish
            stop_loss = entry_price + stop_distance
            target = entry_price - target_distance
            
            trade_strike = nearest_strike
            option_type = "CE"
            action = "SELL"
        
        signal = {
            "action": action,
            "option_type": option_type,
            "strike": trade_strike,
            "entry": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "pattern": confluence['pattern'],
            "strength": confluence['strength'],
            "pcr": confluence['strike_pcr'],
            "reasons": confluence['reasons']
        }
        
        return signal

# ======================== UPSTOX CLIENT (Enhanced) ========================
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
        self.futures_keys = {}  # Store futures key for each symbol
        
    async def create_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    def get_instrument_key(self, symbol: str) -> str:
        """Get correct instrument key for each index"""
        mapping = {
            "NIFTY": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank",
            "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
            "FINNIFTY": "NSE_INDEX|Nifty Fin Service"
        }
        return mapping.get(symbol, f"NSE_EQ|{symbol}")
    
    async def download_instruments(self):
        try:
            logger.info("📡 Downloading instruments...")
            url = UPSTOX_INSTRUMENTS_URL
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"❌ Download failed: {response.status}")
                    return None
                
                content = await response.read()
                json_text = gzip.decompress(content).decode('utf-8')
                instruments = json.loads(json_text)
                
                logger.info(f"✅ Downloaded {len(instruments)} instruments")
                self.instruments_cache = instruments
                
                # Find futures for all symbols for volume data
                now = datetime.now(IST)
                
                for symbol in INDICES:
                    for instrument in instruments:
                        if instrument.get('segment') != 'NSE_FO':
                            continue
                        if instrument.get('instrument_type') != 'FUT':
                            continue
                        if instrument.get('name') != symbol:
                            continue
                        
                        expiry = instrument.get('expiry')
                        if expiry:
                            expiry_dt = datetime.fromtimestamp(expiry / 1000, tz=IST)
                            if expiry_dt > now:
                                self.futures_keys[symbol] = instrument.get('instrument_key')
                                logger.info(f"✅ {symbol} Futures: {self.futures_keys[symbol]}")
                                break
                
                return instruments
                
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return None
    
    async def get_available_expiries(self, symbol: str) -> List[str]:
        """Get available expiry dates for a symbol"""
        try:
            if not self.instruments_cache:
                instruments = await self.download_instruments()
                if not instruments:
                    return []
            else:
                instruments = self.instruments_cache
            
            now = datetime.now(IST)
            expiries_set = set()
            
            for instrument in instruments:
                if instrument.get('segment') != 'NSE_FO':
                    continue
                if instrument.get('instrument_type') not in ['CE', 'PE']:
                    continue
                if instrument.get('name', '') != symbol:
                    continue
                
                expiry_ms = instrument.get('expiry')
                if not expiry_ms:
                    continue
                
                try:
                    expiry_dt = datetime.fromtimestamp(expiry_ms / 1000, tz=IST)
                    if expiry_dt > now:
                        expiry_str = expiry_dt.strftime('%Y-%m-%d')
                        expiries_set.add(expiry_str)
                except:
                    continue
            
            if expiries_set:
                expiries = sorted(list(expiries_set))
                return expiries
            return []
                
        except Exception as e:
            logger.error(f"❌ Error getting expiries for {symbol}: {e}")
            return []
    
    def get_nearest_expiry_for_symbol(self, symbol: str, expiries: List[str]) -> Optional[str]:
        """Get nearest expiry based on symbol's expiry day preference"""
        if not expiries:
            return None
        
        now = datetime.now(IST)
        preferred_weekday = EXPIRY_DAYS.get(symbol, 1)  # Default to Tuesday
        
        # Find nearest expiry matching the preferred weekday
        for expiry_str in expiries:
            try:
                expiry_dt = datetime.strptime(expiry_str, '%Y-%m-%d')
                expiry_dt = IST.localize(expiry_dt)
                
                # Check if this expiry is in the future and matches preferred weekday
                if expiry_dt > now and expiry_dt.weekday() == preferred_weekday:
                    return expiry_str
            except:
                continue
        
        # Fallback: return nearest future expiry
        return expiries[0] if expiries else None
    
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
            return {"ltp": 0.0, "volume": 0, "oi": 0}
    
    async def get_ltp(self, instrument_key: str) -> float:
        quote = await self.get_full_quote(instrument_key)
        return quote["ltp"]
    
    async def get_historical_candles_combined(self, instrument_key: str) -> pd.DataFrame:
        """Get intraday candles only (indices don't support historical API)"""
        try:
            all_candles = []
            
            # For INDEX instruments, ONLY use intraday API
            # Historical API doesn't work for NSE_INDEX instruments
            
            # Get 5-minute intraday candles using V3 API
            url_intraday = f"{UPSTOX_API_V3_URL}/historical-candle/intraday/{instrument_key}/minutes/5"
            
            async with self.session.get(url_intraday) as response:
                data = await response.json()
                
                if data.get("status") == "success" and data.get("data", {}).get("candles"):
                    intraday_candles = data["data"]["candles"]
                    
                    for candle in intraday_candles:
                        all_candles.append({
                            'timestamp': pd.to_datetime(candle[0]),
                            'open': candle[1],
                            'high': candle[2],
                            'low': candle[3],
                            'close': candle[4],
                            'volume': candle[5] if len(candle) > 5 else 0,
                            'oi': candle[6] if len(candle) > 6 else 0
                        })
                    
                    logger.info(f"✅ Fetched {len(intraday_candles)} intraday candles")
                else:
                    logger.warning(f"⚠️ No candle data in response: {data}")
            
            if not all_candles:
                logger.warning(f"⚠️ No candles found for {instrument_key}")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_candles)
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
            
            logger.info(f"✅ Loaded {len(df)} candles")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching candles: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    async def get_futures_volume_map(self, candles_df: pd.DataFrame, symbol: str) -> Dict:
        """Get volume data from futures for each candle timestamp"""
        volume_map = {}
        
        futures_key = self.futures_keys.get(symbol)
        if not futures_key:
            logger.warning(f"⚠️ No futures key for {symbol}")
            return volume_map
        
        try:
            # Use V3 API with minutes/5 format
            url = f"{UPSTOX_API_V3_URL}/historical-candle/intraday/{futures_key}/minutes/5"
            
            async with self.session.get(url) as response:
                data = await response.json()
                
                if data.get("status") == "success" and data.get("data", {}).get("candles"):
                    futures_candles = data["data"]["candles"]
                    
                    for candle in futures_candles:
                        timestamp = pd.to_datetime(candle[0])
                        volume = candle[5] if len(candle) > 5 else 0
                        volume_map[timestamp] = volume
            
            logger.info(f"✅ Fetched volume for {len(volume_map)} candles from {symbol} futures")
            
        except Exception as e:
            logger.error(f"⚠️ Futures volume fetch failed for {symbol}: {e}")
        
        return volume_map
    
    async def get_option_contracts(self, symbol: str, expiry: str) -> List[Dict]:
        try:
            instrument_key = self.get_instrument_key(symbol)
            url = f"{UPSTOX_API_URL}/option/contract"
            
            params = {
                "instrument_key": instrument_key,
                "expiry_date": expiry
            }
            
            async with self.session.get(url, params=params) as response:
                response_text = await response.text()
                data = json.loads(response_text)
                
                if data.get("status") == "success":
                    contracts = data.get("data", [])
                    if contracts:
                        logger.info(f"✅ Fetched {len(contracts)} option contracts for {symbol}")
                        return contracts
                    return []
                return []
                    
        except Exception as e:
            logger.error(f"❌ Error fetching contracts for {symbol}: {e}")
            return []

# ======================== OPTION ANALYZER (Enhanced) ========================
class OptionAnalyzer:
    def __init__(self, client: UpstoxClient):
        self.client = client
    
    def get_strike_interval(self, symbol: str) -> int:
        return STRIKE_INTERVALS.get(symbol, 50)
    
    async def filter_atm_strikes(self, contracts: List[Dict], current_price: float, symbol: str) -> Dict:
        interval = self.get_strike_interval(symbol)
        atm = round(current_price / interval) * interval
        min_strike = atm - (ATM_RANGE * interval)
        max_strike = atm + (ATM_RANGE * interval)
        
        logger.info(f"🎯 ATM: {atm}, Range: {min_strike} to {max_strike}")
        
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
                    "ltp": 0, "oi": 0, "volume": 0
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
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            logger.info(f"\n📊 Analyzing {symbol}...")
            
            instrument_key = self.client.get_instrument_key(symbol)
            current_price = await self.client.get_ltp(instrument_key)
            
            if current_price == 0:
                logger.warning(f"⚠️ Could not fetch price for {symbol}")
                return None
            
            logger.info(f"💰 {symbol} Spot: ₹{current_price:,.2f}")
            
            # Get 200+ candles
            candles = await self.client.get_historical_candles_combined(instrument_key)
            
            if candles.empty:
                logger.warning(f"⚠️ No candle data for {symbol}")
                return None
            
            logger.info(f"📈 Loaded {len(candles)} candles for {symbol}")
            
            # Get futures volume
            volume_map = await self.client.get_futures_volume_map(candles, symbol)
            
            # Detect patterns with volume
            patterns = CandlestickPatterns.detect_patterns_with_volume(candles, volume_map)
            logger.info(f"🎯 Found {len(patterns)} patterns in {symbol}")
            
            # Identify S/R levels
            sr_levels = SupportResistance.identify_levels(candles, current_price)
            logger.info(f"📊 {symbol} Support: {sr_levels['support']}")
            logger.info(f"📊 {symbol} Resistance: {sr_levels['resistance']}")
            
            # Get expiry
            expiries = await self.client.get_available_expiries(symbol)
            if not expiries:
                logger.warning(f"⚠️ No expiries found for {symbol}")
                return None
            
            # Get nearest expiry based on symbol's preferred day
            expiry = self.client.get_nearest_expiry_for_symbol(symbol, expiries)
            
            if not expiry:
                logger.warning(f"⚠️ No suitable expiry found for {symbol}")
                return None
            
            contracts = await self.client.get_option_contracts(symbol, expiry)
            
            if not contracts:
                logger.warning(f"⚠️ No option contracts found for {symbol}")
                return None
            
            contracts_data = await self.filter_atm_strikes(contracts, current_price, symbol)
            
            if not contracts_data["strikes"]:
                logger.warning(f"⚠️ No strikes in ATM range for {symbol}")
                return None
            
            await self.fetch_option_prices(contracts_data)
            
            # Calculate PCR per strike
            for strike in contracts_data["strikes"]:
                ce = contracts_data["ce"].get(strike)
                pe = contracts_data["pe"].get(strike)
                
                if ce and pe:
                    strike_pcr = pe["oi"] / ce["oi"] if ce["oi"] > 0 else 0
                    ce["pcr"] = strike_pcr
                    pe["pcr"] = strike_pcr
            
            # Prepare strike data
            strike_data = {}
            for strike in contracts_data["strikes"]:
                ce = contracts_data["ce"].get(strike, {})
                pe = contracts_data["pe"].get(strike, {})
                
                strike_data[strike] = {
                    "ce_oi": ce.get("oi", 0),
                    "pe_oi": pe.get("oi", 0),
                    "ce_ltp": ce.get("ltp", 0),
                    "pe_ltp": pe.get("ltp", 0),
                    "pcr": ce.get("pcr", 0)
                }
            
            ce_data = []
            pe_data = []
            
            for s in contracts_data["strikes"]:
                ce = contracts_data["ce"].get(s, {"strike": s, "ltp": 0, "oi": 0, "volume": 0, "pcr": 0})
                pe = contracts_data["pe"].get(s, {"strike": s, "ltp": 0, "oi": 0, "volume": 0, "pcr": 0})
                ce_data.append(ce)
                pe_data.append(pe)
            
            atm_strike = round(current_price / self.get_strike_interval(symbol)) * self.get_strike_interval(symbol)
            
            # ✅ CONFLUENCE ANALYSIS
            confluences = []
            trade_signals = []
            
            for pattern in patterns[-10:]:  # Last 10 patterns
                confluence = ConfluenceAnalyzer.analyze_pattern_confluence(
                    pattern, sr_levels, strike_data, atm_strike, symbol
                )
                
                if confluence:
                    confluences.append(confluence)
                    
                    # Generate trade signal
                    signal = ConfluenceAnalyzer.generate_trade_signal(confluence, atm_strike, symbol)
                    if signal:
                        trade_signals.append(signal)
            
            logger.info(f"✅ {symbol}: Found {len(confluences)} confluences, {len(trade_signals)} trade signals")
            
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
                "atm_strike": atm_strike,
                "confluences": confluences,
                "trade_signals": trade_signals,
                "volume_map": volume_map,
                "lot_size": LOT_SIZES.get(symbol, 25)
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

# ======================== IMPROVED CHART GENERATOR ========================
class ChartGenerator:
    @staticmethod
    def create_combined_chart(analysis: Dict) -> BytesIO:
        """Create enhanced chart with improved formatting - 26x18 inches"""
        symbol = analysis["symbol"]
        candles = analysis["candles"]
        current_price = analysis["current_price"]
        patterns = analysis.get("patterns", [])
        sr_levels = analysis.get("sr_levels", {"support": [], "resistance": []})
        confluences = analysis.get("confluences", [])
        trade_signals = analysis.get("trade_signals", [])
        lot_size = analysis.get("lot_size", 25)
        
        # Get current time
        now_time = datetime.now(IST).strftime('%H:%M:%S IST')
        
        # Calculate overall PCR and trend
        total_ce_oi = sum(d["oi"] for d in analysis["ce_data"])
        total_pe_oi = sum(d["oi"] for d in analysis["pe_data"])
        overall_pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        if overall_pcr > 1.3:
            trend_emoji = "🟢"
            trend_text = "BULLISH"
            trend_color = "#26a69a"
        elif overall_pcr > 1.0:
            trend_emoji = "🟢"
            trend_text = "Slight Bullish"
            trend_color = "#4db8a3"
        elif overall_pcr >= 0.9:
            trend_emoji = "⚪"
            trend_text = "NEUTRAL"
            trend_color = "#757575"
        elif overall_pcr >= 0.7:
            trend_emoji = "🔴"
            trend_text = "Slight Bearish"
            trend_color = "#ef9a9a"
        else:
            trend_emoji = "🔴"
            trend_text = "BEARISH"
            trend_color = "#ef5350"
        
        # ✅ INCREASED SIZE: 26x18 inches for better spacing
        fig = plt.figure(figsize=(26, 18), facecolor='white')
        gs = GridSpec(5, 2, height_ratios=[2.8, 0.6, 0.9, 0.9, 1.2], width_ratios=[1.5, 1], hspace=0.5, wspace=0.3)
        
        # ========== CANDLESTICK CHART (LARGER) ==========
        ax1 = fig.add_subplot(gs[0, :])  # Span both columns
        
        mc = mpf.make_marketcolors(
            up='#26a69a', down='#ef5350',
            edge='inherit',
            wick={'up': '#26a69a', 'down': '#ef5350'},
            volume='in', alpha=0.9
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc, gridstyle='--', gridcolor='#e0e0e0',
            facecolor='white', figcolor='white', y_on_right=False
        )
        
        # Show last 100 candles on chart
        candles_display = candles.tail(100)
        
        mpf.plot(
            candles_display, type='candle', style=s, ax=ax1,
            volume=False, show_nontrading=False
        )
        
        # Draw Support Levels with better positioning
        for i, support in enumerate(sr_levels['support']):
            y_offset = 0.05 + (i * 0.08)  # Spread out labels vertically
            ax1.axhline(y=support, color='green', linestyle='--', linewidth=2.5, alpha=0.7)
            ax1.text(0.97, support, f'Support ₹{support:.0f}  ', 
                    transform=ax1.get_yaxis_transform(), 
                    color='white', fontsize=12, fontweight='bold', va='center', ha='right',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='green', edgecolor='darkgreen', alpha=0.9, linewidth=2))
        
        # Draw Resistance Levels with better positioning
        for i, resistance in enumerate(sr_levels['resistance']):
            y_offset = 0.05 + (i * 0.08)
            ax1.axhline(y=resistance, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
            ax1.text(0.97, resistance, f'Resistance ₹{resistance:.0f}  ', 
                    transform=ax1.get_yaxis_transform(), 
                    color='white', fontsize=12, fontweight='bold', va='center', ha='right',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='red', edgecolor='darkred', alpha=0.9, linewidth=2))
        
        # Mark Candlestick Patterns with shorter names
        pattern_short_names = {
            "🔨 HAMMER": "🔨 HAM",
            "⭐ SHOOTING STAR": "⭐ STAR",
            "🟢 BULLISH ENGULFING": "🟢 BULL ENG",
            "🔴 BEARISH ENGULFING": "🔴 BEAR ENG",
            "🌅 MORNING STAR": "🌅 MORN",
            "🌆 EVENING STAR": "🌆 EVE",
            "✖️ DOJI": "✖️ DOJI"
        }
        
        for pattern in patterns[-8:]:
            idx = pattern['index']
            if idx < len(candles):
                candle_time = candles.index[idx]
                
                if candle_time not in candles_display.index:
                    continue
                
                candle_high = candles.iloc[idx]['high']
                
                color = 'green' if pattern['type'] == 'bullish' else ('red' if pattern['type'] == 'bearish' else 'blue')
                marker = '▲' if pattern['type'] == 'bullish' else ('▼' if pattern['type'] == 'bearish' else '●')
                
                vol_text = "📊" if pattern.get('high_volume') else ""
                short_name = pattern_short_names.get(pattern['pattern'], pattern['pattern'])
                
                ax1.annotate(
                    f"{short_name} {vol_text}", 
                    xy=(candle_time, candle_high),
                    xytext=(0, 30), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, alpha=0.95),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color=color, lw=2)
                )
        
        # Enhanced title with trend and time
        title_text = f"{symbol} - 5min | Spot: ₹{current_price:,.2f} | {trend_emoji} {trend_text} (PCR: {overall_pcr:.2f}) | Expiry: {analysis['expiry']} | Lot: {lot_size} | ⏰ {now_time}"
        ax1.set_title(title_text, fontsize=18, fontweight='bold', pad=20, color='#1a1a1a')
        ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # ========== TRADE SIGNALS (LEFT COLUMN) ==========
        ax_signals = fig.add_subplot(gs[1, 0])
        ax_signals.axis('off')
        
        signals_text = f"🎯 HIGH PROBABILITY SETUPS\n" + "="*60 + "\n"
        
        if trade_signals:
            for i, sig in enumerate(trade_signals[:2], 1):  # Top 2 only
                action_emoji = "📗" if sig['action'] == "BUY" else "📕"
                signals_text += f"\n{action_emoji} #{i}: {sig['action']} {sig['strike']} {sig['option_type']} @ ₹{sig['entry']:.0f}\n"
                signals_text += f"   ⛔ SL: ₹{sig['stop_loss']:.0f}  |  🎯 TGT: ₹{sig['target']:.0f}  |  💪 {sig['strength']}\n"
                signals_text += f"   📊 {sig['pattern'][:30]} | PCR: {sig['pcr']:.2f}\n"
        else:
            signals_text += "\n⚠️  No high-probability setups found.\n⏳ Wait for pattern + PCR confluence.\n"
        
        ax_signals.text(0.02, 0.98, signals_text, transform=ax_signals.transAxes,
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.8', facecolor='#fff3cd', edgecolor='#ff9800', alpha=0.95, linewidth=2.5))
        
        # ========== KEY LEVELS (RIGHT COLUMN) ==========
        ax_levels = fig.add_subplot(gs[1, 1])
        ax_levels.axis('off')
        
        interval = STRIKE_INTERVALS.get(symbol, 50)
        levels_text = f"📍 KEY LEVELS (PCR Analysis)\n" + "="*40 + "\n"
        
        # Support levels
        levels_text += "\n🟢 SUPPORT ZONES:\n"
        for support in sr_levels['support'][:2]:  # Top 2 only
            nearest_strike = round(support / interval) * interval
            strike_info = next((d for d in analysis['ce_data'] if d['strike'] == nearest_strike), None)
            pcr = strike_info.get('pcr', 0) if strike_info else 0
            levels_text += f"   ₹{support:.0f} → Strike {nearest_strike} (PCR {pcr:.2f})\n"
        
        # Resistance levels
        levels_text += "\n🔴 RESISTANCE ZONES:\n"
        for resistance in sr_levels['resistance'][:2]:  # Top 2 only
            nearest_strike = round(resistance / interval) * interval
            strike_info = next((d for d in analysis['ce_data'] if d['strike'] == nearest_strike), None)
            pcr = strike_info.get('pcr', 0) if strike_info else 0
            levels_text += f"   ₹{resistance:.0f} → Strike {nearest_strike} (PCR {pcr:.2f})\n"
        
        ax_levels.text(0.02, 0.98, levels_text, transform=ax_levels.transAxes,
                      fontsize=11, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round,pad=0.8', facecolor='#e3f2fd', edgecolor='#2196f3', alpha=0.95, linewidth=2.5))
        
        # ========== PATTERN ANALYSIS (LEFT COLUMN) ==========
        ax_patterns = fig.add_subplot(gs[2, 0])
        ax_patterns.axis('off')
        
        pattern_text = f"📊 RECENT PATTERNS\n" + "="*60 + "\n"
        
        if confluences:
            for conf in confluences[-3:]:  # Last 3 only
                time_str = conf['time'].strftime('%H:%M')
                strength_emoji = "💪💪" if conf['strength'] == "STRONG" else "💪"
                pattern_text += f"\n⏰ {time_str} | {conf['pattern'][:25]} @ ₹{conf['price']:.0f}\n"
                pattern_text += f"   Strike: ₹{conf['nearest_strike']} | PCR: {conf['strike_pcr']:.2f} | {strength_emoji} {conf['strength']}\n"
        else:
            pattern_text += "\n⚠️ No recent patterns with strong confluence.\n"
        
        ax_patterns.text(0.02, 0.98, pattern_text, transform=ax_patterns.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.7', facecolor='#f3e5f5', edgecolor='#9c27b0', alpha=0.9, linewidth=2))
        
        # ========== MARKET SENTIMENT (RIGHT COLUMN) ==========
        ax_sentiment = fig.add_subplot(gs[2, 1])
        ax_sentiment.axis('off')
        
        sentiment_text = f"📈 MARKET SENTIMENT\n" + "="*40 + "\n\n"
        sentiment_text += f"Overall PCR: {overall_pcr:.3f}\n"
        sentiment_text += f"Bias: {trend_emoji} {trend_text}\n\n"
        sentiment_text += f"Total CE OI: {total_ce_oi:,.0f}\n"
        sentiment_text += f"Total PE OI: {total_pe_oi:,.0f}\n\n"
        
        if overall_pcr > 1.2:
            sentiment_text += "💡 High PCR indicates strong\n   PUT writing → Bullish bias"
        elif overall_pcr < 0.8:
            sentiment_text += "💡 Low PCR indicates strong\n   CALL writing → Bearish bias"
        else:
            sentiment_text += "💡 Balanced PCR indicates\n   neutral market positioning"
        
        ax_sentiment.text(0.02, 0.98, sentiment_text, transform=ax_sentiment.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.7', facecolor='#fff9c4', edgecolor='#fbc02d', alpha=0.9, linewidth=2))
        
        # ========== OPTION CHAIN TABLE (FULL WIDTH) ==========
        ax_table = fig.add_subplot(gs[3:, :])
        ax_table.axis('tight')
        ax_table.axis('off')
        
        # Compact table header
        table_data = [["Strike", "PE OI", "CE OI", "PCR", "CE ₹", "PE ₹", "Signal"]]
        
        atm_strike = analysis['atm_strike']
        
        for i, strike in enumerate(analysis["strikes"]):
            ce = analysis["ce_data"][i]
            pe = analysis["pe_data"][i]
            
            pcr = pe.get("pcr", 0)
            
            # Shorter signal names
            if pcr > 2.0:
                signal = "🟢🟢 STR SUP"
            elif pcr > 1.5:
                signal = "🟢 Support"
            elif pcr > 1.1:
                signal = "⚪ Neut+"
            elif pcr >= 0.9:
                signal = "⚪ Balance"
            elif pcr >= 0.6:
                signal = "🔴 Resist"
            else:
                signal = "🔴🔴 STR RES"
            
            # Format numbers compactly
            pe_oi_str = f"{pe['oi']/1000:.1f}K" if pe['oi'] < 1000000 else f"{pe['oi']/1000000:.2f}M"
            ce_oi_str = f"{ce['oi']/1000:.1f}K" if ce['oi'] < 1000000 else f"{ce['oi']/1000000:.2f}M"
            
            row = [
                f"₹{strike:,.0f}{'*' if strike == atm_strike else ''}",
                pe_oi_str,
                ce_oi_str,
                f"{pcr:.2f}",
                f"₹{ce['ltp']:.1f}",
                f"₹{pe['ltp']:.1f}",
                signal
            ]
            table_data.append(row)
        
        # Overall row
        overall_pe = f"{total_pe_oi/1000000:.2f}M"
        overall_ce = f"{total_ce_oi/1000000:.2f}M"
        
        if overall_pcr > 1.3:
            overall_signal = f"{trend_emoji} {trend_text}"
        elif overall_pcr >= 0.9:
            overall_signal = f"{trend_emoji} {trend_text}"
        else:
            overall_signal = f"{trend_emoji} {trend_text}"
        
        table_data.append([
            "OVERALL",
            overall_pe,
            overall_ce,
            f"{overall_pcr:.2f}",
            "", "",
            overall_signal
        ])
        
        table = ax_table.table(
            cellText=table_data, loc='center', cellLoc='center',
            colWidths=[0.14, 0.12, 0.12, 0.10, 0.12, 0.12, 0.16]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 3.2)
        
        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#37474f')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
        
        # Style summary row
        summary_row = len(table_data) - 1
        for i in range(7):
            table[(summary_row, i)].set_facecolor('#ffd54f')
            table[(summary_row, i)].set_text_props(weight='bold', fontsize=12)
        
        # Highlight ATM row
        for i, strike in enumerate(analysis["strikes"], 1):
            if strike == atm_strike:
                for j in range(7):
                    table[(i, j)].set_facecolor('#bbdefb')
                    table[(i, j)].set_text_props(weight='bold')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=200, facecolor='white', bbox_inches='tight')
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
            total_ce_oi = sum(d["oi"] for d in analysis["ce_data"])
            total_pe_oi = sum(d["oi"] for d in analysis["pe_data"])
            pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            
            if pcr > 1.5:
                bias = "🟢 BULLISH"
            elif pcr > 1.1:
                bias = "🟢 Slightly Bullish"
            elif pcr >= 0.9:
                bias = "⚪ NEUTRAL"
            elif pcr >= 0.7:
                bias = "🔴 Slightly Bearish"
            else:
                bias = "🔴 BEARISH"
            
            # Trade signals summary
            trade_signals = analysis.get('trade_signals', [])
            signals_text = ""
            
            if trade_signals:
                signals_text = "\n\n🎯 TRADE SIGNALS:\n"
                for sig in trade_signals[:2]:
                    signals_text += f"• {sig['action']} {sig['strike']} {sig['option_type']} @ ₹{sig['entry']:.0f}\n"
                    signals_text += f"  SL: ₹{sig['stop_loss']:.0f} | TGT: ₹{sig['target']:.0f}\n"
            
            lot_size = analysis.get('lot_size', 25)
            expiry_day = EXPIRY_DAYS.get(symbol, 1)
            expiry_day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][expiry_day]
            
            caption = f"""📊 {symbol} Enhanced Analysis

💰 Spot: ₹{analysis['current_price']:,.2f}
📅 Expiry: {analysis['expiry']} ({expiry_day_name})
📦 Lot Size: {lot_size}

📈 CE OI: {total_ce_oi:,.0f}
📉 PE OI: {total_pe_oi:,.0f}
🔄 PCR: {pcr:.3f}

{bias}{signals_text}

⏰ {datetime.now(IST).strftime('%d-%b %H:%M IST')}

✅ Multi-Index Bot v5.0"""
            
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=chart_buffer,
                caption=caption
            )
            
            logger.info(f"✅ Alert sent for {symbol}")
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error for {symbol}: {e}")
        except Exception as e:
            logger.error(f"❌ Error sending alert for {symbol}: {e}")

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
        
        logger.info("\n" + "="*70)
        logger.info(f"🔍 MULTI-INDEX ANALYSIS CYCLE - {now_time.strftime('%H:%M:%S IST')}")
        logger.info("="*70)
        
        for symbol in INDICES:
            try:
                analysis = await self.analyzer.analyze_symbol(symbol)
                
                if analysis:
                    chart = self.chart_gen.create_combined_chart(analysis)
                    await self.alerter.send_chart(chart, symbol, analysis)
                    logger.info(f"✅ {symbol} complete\n")
                else:
                    logger.warning(f"⚠️ {symbol} analysis failed\n")
                
                await asyncio.sleep(3)  # Delay between indices
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}\n")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("="*70)
        logger.info("✅ CYCLE COMPLETE")
        logger.info("="*70 + "\n")
    
    async def run(self):
        current_time = datetime.now(IST)
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        
        print("\n" + "="*70)
        print("🚀 ENHANCED MULTI-INDEX UPSTOX BOT v5.0", flush=True)
        print("="*70)
        print(f"📅 {current_time.strftime('%d-%b-%Y %A')}", flush=True)
        print(f"🕐 {current_time.strftime('%H:%M:%S IST')}", flush=True)
        print(f"📊 Market: {market_status}", flush=True)
        print(f"⏱️  Interval: 5 minutes", flush=True)
        print(f"📈 Indices: {', '.join(INDICES)}", flush=True)
        print(f"✅ Pattern-Strike Link | Confluence | Trade Signals", flush=True)
        print(f"✅ Auto Expiry Selection (All on Tuesday)", flush=True)
        print(f"📊 200+ Candles | Futures Volume | Chart: 24x16", flush=True)
        print("="*70 + "\n", flush=True)
        
        await self.client.create_session()
        
        try:
            # Download instruments once at start
            await self.client.download_instruments()
            
            while True:
                try:
                    await self.process_symbols()
                    
                    next_run = datetime.now(IST) + timedelta(seconds=ANALYSIS_INTERVAL)
                    logger.info(f"⏰ Next cycle: {next_run.strftime('%H:%M:%S')}\n")
                    
                    await asyncio.sleep(ANALYSIS_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}")
                    await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
        
        finally:
            await self.client.close_session()
            logger.info("👋 Session closed")

# ======================== ENTRY POINT ========================
if __name__ == "__main__":
    bot = UpstoxOptionsBot()
    asyncio.run(bot.run())
