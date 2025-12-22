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
UPSTOX_API_V3_URL = "https://api.upstox.com/v3"
UPSTOX_INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz"
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# Trading params
ANALYSIS_INTERVAL = 5 * 60
CANDLES_COUNT = 50
ATM_RANGE = 5

# Market hours (IST)
MARKET_START = dt_time(9, 15)
MARKET_END = dt_time(15, 30)
IST = pytz.timezone('Asia/Kolkata')

INDICES = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================== CANDLESTICK PATTERN DETECTOR ========================
class CandlestickPatterns:
    @staticmethod
    def is_hammer(row, prev_row=None) -> Tuple[bool, str]:
        """Hammer - Bullish reversal"""
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, ""
        
        # Hammer: small body, long lower wick, small upper wick
        if (lower_wick > body * 2 and 
            upper_wick < body * 0.3 and 
            body < total_range * 0.3):
            return True, "🔨 HAMMER (Bullish Reversal)"
        return False, ""
    
    @staticmethod
    def is_shooting_star(row, prev_row=None) -> Tuple[bool, str]:
        """Shooting Star - Bearish reversal"""
        body = abs(row['close'] - row['open'])
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, ""
        
        # Shooting star: small body, long upper wick, small lower wick
        if (upper_wick > body * 2 and 
            lower_wick < body * 0.3 and 
            body < total_range * 0.3):
            return True, "⭐ SHOOTING STAR (Bearish Reversal)"
        return False, ""
    
    @staticmethod
    def is_doji(row, prev_row=None) -> Tuple[bool, str]:
        """Doji - Indecision"""
        body = abs(row['close'] - row['open'])
        total_range = row['high'] - row['low']
        
        if total_range == 0:
            return False, ""
        
        # Doji: very small body
        if body < total_range * 0.1:
            return True, "✖️ DOJI (Indecision)"
        return False, ""
    
    @staticmethod
    def is_engulfing(row, prev_row) -> Tuple[bool, str]:
        """Bullish/Bearish Engulfing"""
        if prev_row is None:
            return False, ""
        
        curr_body = abs(row['close'] - row['open'])
        prev_body = abs(prev_row['close'] - prev_row['open'])
        
        # Bullish engulfing: green candle engulfs previous red
        if (row['close'] > row['open'] and 
            prev_row['close'] < prev_row['open'] and
            row['open'] <= prev_row['close'] and
            row['close'] >= prev_row['open'] and
            curr_body > prev_body * 1.2):
            return True, "🟢 BULLISH ENGULFING (Strong Buy)"
        
        # Bearish engulfing: red candle engulfs previous green
        if (row['close'] < row['open'] and 
            prev_row['close'] > prev_row['open'] and
            row['open'] >= prev_row['close'] and
            row['close'] <= prev_row['open'] and
            curr_body > prev_body * 1.2):
            return True, "🔴 BEARISH ENGULFING (Strong Sell)"
        
        return False, ""
    
    @staticmethod
    def is_morning_star(df, idx) -> Tuple[bool, str]:
        """Morning Star - Bullish reversal (3 candle pattern)"""
        if idx < 2:
            return False, ""
        
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        
        # First: long red, Second: small body, Third: long green
        first_red = first['close'] < first['open']
        second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
        third_green = third['close'] > third['open']
        third_closes_high = third['close'] > (first['open'] + first['close']) / 2
        
        if first_red and second_small and third_green and third_closes_high:
            return True, "🌅 MORNING STAR (Bullish Reversal)"
        return False, ""
    
    @staticmethod
    def is_evening_star(df, idx) -> Tuple[bool, str]:
        """Evening Star - Bearish reversal (3 candle pattern)"""
        if idx < 2:
            return False, ""
        
        first = df.iloc[idx-2]
        second = df.iloc[idx-1]
        third = df.iloc[idx]
        
        # First: long green, Second: small body, Third: long red
        first_green = first['close'] > first['open']
        second_small = abs(second['close'] - second['open']) < abs(first['close'] - first['open']) * 0.3
        third_red = third['close'] < third['open']
        third_closes_low = third['close'] < (first['open'] + first['close']) / 2
        
        if first_green and second_small and third_red and third_closes_low:
            return True, "🌆 EVENING STAR (Bearish Reversal)"
        return False, ""
    
    @staticmethod
    def detect_all_patterns(df) -> List[Dict]:
        """Detect all patterns in dataframe"""
        patterns = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1] if i > 0 else None
            
            # Single candle patterns
            is_pat, name = CandlestickPatterns.is_hammer(row, prev_row)
            if is_pat:
                patterns.append({"index": i, "time": row.name, "pattern": name, "type": "bullish"})
                continue
            
            is_pat, name = CandlestickPatterns.is_shooting_star(row, prev_row)
            if is_pat:
                patterns.append({"index": i, "time": row.name, "pattern": name, "type": "bearish"})
                continue
            
            is_pat, name = CandlestickPatterns.is_doji(row, prev_row)
            if is_pat:
                patterns.append({"index": i, "time": row.name, "pattern": name, "type": "neutral"})
                continue
            
            # Two candle patterns
            if prev_row is not None:
                is_pat, name = CandlestickPatterns.is_engulfing(row, prev_row)
                if is_pat:
                    pat_type = "bullish" if "BULLISH" in name else "bearish"
                    patterns.append({"index": i, "time": row.name, "pattern": name, "type": pat_type})
                    continue
            
            # Three candle patterns
            is_pat, name = CandlestickPatterns.is_morning_star(df, i)
            if is_pat:
                patterns.append({"index": i, "time": row.name, "pattern": name, "type": "bullish"})
                continue
            
            is_pat, name = CandlestickPatterns.is_evening_star(df, i)
            if is_pat:
                patterns.append({"index": i, "time": row.name, "pattern": name, "type": "bearish"})
        
        return patterns

# ======================== SUPPORT & RESISTANCE DETECTOR ========================
class SupportResistance:
    @staticmethod
    def find_pivot_points(df, window=3) -> Tuple[List, List]:
        """Find pivot highs and lows"""
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # Check if current high is highest in window
            if df.iloc[i]['high'] == df.iloc[i-window:i+window+1]['high'].max():
                highs.append((i, df.iloc[i]['high']))
            
            # Check if current low is lowest in window
            if df.iloc[i]['low'] == df.iloc[i-window:i+window+1]['low'].min():
                lows.append((i, df.iloc[i]['low']))
        
        return highs, lows
    
    @staticmethod
    def cluster_levels(levels, tolerance=0.005) -> List[float]:
        """Cluster nearby levels together"""
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
        
        # Extract prices
        high_prices = [h[1] for h in highs]
        low_prices = [l[1] for l in lows]
        
        # Cluster levels
        resistance_levels = SupportResistance.cluster_levels(high_prices)
        support_levels = SupportResistance.cluster_levels(low_prices)
        
        # Filter: resistance above current, support below current
        resistance_levels = [r for r in resistance_levels if r > current_price]
        support_levels = [s for s in support_levels if s < current_price]
        
        # Sort and get top 3
        resistance_levels = sorted(resistance_levels)[:3]
        support_levels = sorted(support_levels, reverse=True)[:3]
        
        return {
            "resistance": resistance_levels,
            "support": support_levels
        }

# ======================== UPSTOX API CLIENT (with S/R analysis) ========================
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
    
    async def download_instruments(self):
        try:
            logger.info("📡 Downloading instruments from Upstox CDN...")
            url = UPSTOX_INSTRUMENTS_URL
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"❌ Instruments download failed: {response.status}")
                    return None
                
                content = await response.read()
                json_text = gzip.decompress(content).decode('utf-8')
                instruments = json.loads(json_text)
                
                logger.info(f"✅ Downloaded {len(instruments)} instruments")
                self.instruments_cache = instruments
                return instruments
                
        except Exception as e:
            logger.error(f"❌ Error downloading instruments: {e}")
            return None
    
    async def get_available_expiries(self, symbol: str) -> List[str]:
        try:
            if not self.instruments_cache:
                instruments = await self.download_instruments()
                if not instruments:
                    return []
            else:
                instruments = self.instruments_cache
            
            logger.info(f"   📅 Finding expiries for {symbol}...")
            
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
                segment = instrument.get('segment')
                if segment != 'NSE_FO':
                    continue
                
                inst_type = instrument.get('instrument_type')
                if inst_type not in ['CE', 'PE']:
                    continue
                
                name = instrument.get('name', '')
                if name != instrument_name:
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
                logger.info(f"   ✅ Found {len(expiries)} future expiries")
                return expiries
            else:
                logger.warning(f"   ⚠️ No future expiries found for {symbol}")
                return []
                
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
            logger.error(f"❌ Error fetching quote: {e}")
            return {"ltp": 0.0, "volume": 0, "oi": 0}
    
    async def get_ltp(self, instrument_key: str) -> float:
        quote = await self.get_full_quote(instrument_key)
        return quote["ltp"]
    
    async def get_historical_candles(self, instrument_key: str, count: int = 50) -> pd.DataFrame:
        try:
            url = f"{UPSTOX_API_V3_URL}/historical-candle/intraday/{instrument_key}/minutes/5"
            
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
            logger.error(f"❌ Error fetching candles: {e}")
            return pd.DataFrame()
    
    async def get_option_contracts(self, symbol: str, expiry: str) -> List[Dict]:
        try:
            instrument_key = self.get_instrument_key(symbol)
            url = f"{UPSTOX_API_URL}/option/contract"
            
            params = {
                "instrument_key": instrument_key,
                "expiry_date": expiry
            }
            
            logger.debug(f"   📡 Fetching option contracts")
            
            async with self.session.get(url, params=params) as response:
                response_text = await response.text()
                data = json.loads(response_text)
                
                if data.get("status") == "success":
                    contracts = data.get("data", [])
                    
                    if contracts:
                        logger.info(f"   ✅ Fetched {len(contracts)} option contracts")
                        return contracts
                    else:
                        logger.warning(f"   ⚠️ No contracts for expiry {expiry}")
                        return []
                else:
                    logger.error(f"   ❌ API Error: {data}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error fetching option contracts: {e}")
            return []

# ======================== OPTION CHAIN ANALYZER ========================
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
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            logger.info(f"📊 Analyzing {symbol}...")
            
            instrument_key = self.client.get_instrument_key(symbol)
            current_price = await self.client.get_ltp(instrument_key)
            
            if current_price == 0:
                logger.warning(f"⚠️ Could not fetch price for {symbol}")
                return None
            
            logger.info(f"   💰 LTP: ₹{current_price:,.2f}")
            
            candles = await self.client.get_historical_candles(instrument_key, CANDLES_COUNT)
            
            if candles.empty:
                logger.warning(f"⚠️ No candle data for {symbol}")
                return None
            
            logger.info(f"   📈 Fetched {len(candles)} candles")
            
            # ✅ Detect candlestick patterns
            patterns = CandlestickPatterns.detect_all_patterns(candles)
            logger.info(f"   🎯 Found {len(patterns)} candlestick patterns")
            
            # ✅ Identify support/resistance
            sr_levels = SupportResistance.identify_levels(candles, current_price)
            logger.info(f"   📊 Support levels: {sr_levels['support']}")
            logger.info(f"   📊 Resistance levels: {sr_levels['resistance']}")
            
            expiries = await self.client.get_available_expiries(symbol)
            
            if not expiries:
                logger.warning(f"⚠️ No expiries found for {symbol}")
                return None
            
            contracts = []
            expiry = None
            
            for exp_date in expiries[:3]:
                logger.info(f"   📅 Trying expiry: {exp_date}")
                contracts = await self.client.get_option_contracts(symbol, exp_date)
                
                if contracts:
                    expiry = exp_date
                    logger.info(f"   ✅ Using expiry: {expiry}")
                    break
            
            if not contracts or not expiry:
                logger.warning(f"⚠️ No valid option contracts found for {symbol}")
                return None
            
            contracts_data = await self.filter_atm_strikes(contracts, current_price, symbol)
            
            if not contracts_data["strikes"]:
                logger.warning(f"⚠️ No strikes found in ATM range")
                return None
            
            await self.fetch_option_prices(contracts_data)
            
            for strike in contracts_data["strikes"]:
                ce = contracts_data["ce"].get(strike)
                pe = contracts_data["pe"].get(strike)
                
                if ce and pe:
                    strike_pcr = pe["oi"] / ce["oi"] if ce["oi"] > 0 else 0
                    ce["pcr"] = strike_pcr
                    pe["pcr"] = strike_pcr
            
            ce_data = []
            pe_data = []
            
            for s in contracts_data["strikes"]:
                ce = contracts_data["ce"].get(s, {
                    "strike": s, "ltp": 0, "oi": 0, "volume": 0, "pcr": 0
                })
                pe = contracts_data["pe"].get(s, {
                    "strike": s, "ltp": 0, "oi": 0, "volume": 0, "pcr": 0
                })
                ce_data.append(ce)
                pe_data.append(pe)
            
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
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None

# ======================== ENHANCED CHART GENERATOR ========================
class ChartGenerator:
    @staticmethod
    def create_combined_chart(analysis: Dict) -> BytesIO:
        """Create enhanced chart with S/R and patterns"""
        symbol = analysis["symbol"]
        candles = analysis["candles"]
        current_price = analysis["current_price"]
        patterns = analysis.get("patterns", [])
        sr_levels = analysis.get("sr_levels", {"support": [], "resistance": []})
        
        # ✅ INCREASED SIZE: 20x14 inches
        fig = plt.figure(figsize=(20, 14), facecolor='white')
        gs = GridSpec(3, 1, height_ratios=[2.5, 1, 1], hspace=0.3)
        
        # ========== CANDLESTICK CHART ==========
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
        
        # ✅ Draw Support Levels (Green)
        for support in sr_levels['support']:
            ax1.axhline(y=support, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Support: ₹{support:.0f}')
            ax1.text(0.02, support, f'  Support ₹{support:.0f}', transform=ax1.get_yaxis_transform(), 
                    color='green', fontsize=10, fontweight='bold', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # ✅ Draw Resistance Levels (Red)
        for resistance in sr_levels['resistance']:
            ax1.axhline(y=resistance, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Resistance: ₹{resistance:.0f}')
            ax1.text(0.02, resistance, f'  Resistance ₹{resistance:.0f}', transform=ax1.get_yaxis_transform(), 
                    color='red', fontsize=10, fontweight='bold', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        # ✅ Mark Candlestick Patterns
        for pattern in patterns[-10:]:  # Show last 10 patterns
            idx = pattern['index']
            if idx < len(candles):
                candle_time = candles.index[idx]
                candle_high = candles.iloc[idx]['high']
                
                # Color based on type
                color = 'green' if pattern['type'] == 'bullish' else ('red' if pattern['type'] == 'bearish' else 'blue')
                marker = '▲' if pattern['type'] == 'bullish' else ('▼' if pattern['type'] == 'bearish' else '●')
                
                ax1.annotate(
                    f"{marker} {pattern['pattern']}", 
                    xy=(candle_time, candle_high),
                    xytext=(0, 20),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, alpha=0.9),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color=color)
                )
        
        ax1.set_title(f"{symbol} - 5min Chart | LTP: ₹{current_price:,.2f} | Expiry: {analysis['expiry']}", 
                     fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # ========== PATTERN SUMMARY ==========
        ax_patterns = fig.add_subplot(gs[1])
        ax_patterns.axis('off')
        
        pattern_text = "🎯 DETECTED PATTERNS:\n"
        recent_patterns = patterns[-5:] if len(patterns) > 5 else patterns
        
        if recent_patterns:
            for p in recent_patterns:
                time_str = p['time'].strftime('%H:%M')
                pattern_text += f"  • {time_str} - {p['pattern']}\n"
        else:
            pattern_text += "  No significant patterns detected\n"
        
        # Add S/R info with PCR
        interval = 50 if symbol == "NIFTY" else (100 if symbol == "BANKNIFTY" else 50)
        
        pattern_text += f"\n📊 SUPPORT & RESISTANCE:\n"
        
        for support in sr_levels['support']:
            # Find nearest strike
            nearest_strike = round(support / interval) * interval
            strike_data = next((d for d in analysis['ce_data'] if d['strike'] == nearest_strike), None)
            pcr = strike_data.get('pcr', 0) if strike_data else 0
            
            pattern_text += f"  🟢 Support ₹{support:.0f} (Strike: ₹{nearest_strike}, PCR: {pcr:.2f})\n"
        
        for resistance in sr_levels['resistance']:
            nearest_strike = round(resistance / interval) * interval
            strike_data = next((d for d in analysis['ce_data'] if d['strike'] == nearest_strike), None)
            pcr = strike_data.get('pcr', 0) if strike_data else 0
            
            pattern_text += f"  🔴 Resistance ₹{resistance:.0f} (Strike: ₹{nearest_strike}, PCR: {pcr:.2f})\n"
        
        ax_patterns.text(0.05, 0.95, pattern_text, transform=ax_patterns.transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ========== OPTION CHAIN TABLE ==========
        ax2 = fig.add_subplot(gs[2])
        ax2.axis('tight')
        ax2.axis('off')
        
        table_data = [["Strike", "Put OI", "Call OI", "PCR", "CE LTP", "PE LTP", "Signal", "Action"]]
        
        atm_strike = round(current_price / interval) * interval
        
        for i, strike in enumerate(analysis["strikes"]):
            ce = analysis["ce_data"][i]
            pe = analysis["pe_data"][i]
            
            pcr = pe.get("pcr", 0)
            
            if pcr > 2.5:
                signal = "🟢🟢 STRONG SUPPORT"
                action = "Buy zone"
            elif pcr > 1.5:
                signal = "🟢 Support"
                action = "Bullish"
            elif pcr > 1.1:
                signal = "⚪ Neutral+"
                action = "Slight bull"
            elif pcr >= 0.9:
                signal = "⚪ Balanced"
                action = "Watch"
            elif pcr >= 0.5:
                signal = "🔴 Resistance"
                action = "Bearish"
            else:
                signal = "🔴🔴 STRONG"
                action = "Sell zone"
            
            row = [
                f"₹{strike:,.0f}{'*' if strike == atm_strike else ''}",
                f"{pe['oi']:,}",
                f"{ce['oi']:,}",
                f"{pcr:.2f}",
                f"₹{ce['ltp']:.2f}",
                f"₹{pe['ltp']:.2f}",
                signal,
                action
            ]
            table_data.append(row)
        
        total_ce_oi = sum(d["oi"] for d in analysis["ce_data"])
        total_pe_oi = sum(d["oi"] for d in analysis["pe_data"])
        overall_pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        if overall_pcr > 1.5:
            market_signal = "🟢 BULLISH"
        elif overall_pcr > 1.1:
            market_signal = "🟢 Slight Bull"
        elif overall_pcr >= 0.9:
            market_signal = "⚪ NEUTRAL"
        elif overall_pcr >= 0.7:
            market_signal = "🔴 Slight Bear"
        else:
            market_signal = "🔴 BEARISH"
        
        table_data.append([
            "OVERALL",
            f"{total_pe_oi:,.0f}",
            f"{total_ce_oi:,.0f}",
            f"{overall_pcr:.2f}",
            "",
            "",
            market_signal,
            "See above"
        ])
        
        table = ax2.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.10, 0.12, 0.12, 0.08, 0.10, 0.10, 0.16, 0.12]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        for i in range(8):
            table[(0, i)].set_facecolor('#4a4a4a')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        summary_row = len(table_data) - 1
        for i in range(8):
            table[(summary_row, i)].set_facecolor('#ffd54f')
            table[(summary_row, i)].set_text_props(weight='bold')
        
        for i, strike in enumerate(analysis["strikes"], 1):
            if strike == atm_strike:
                for j in range(8):
                    table[(i, j)].set_facecolor('#e3f2fd')
        
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
                action = "Look for buying opportunities"
            elif pcr > 1.1:
                bias = "🟢 Slightly Bullish"
                action = "Moderate bullish stance"
            elif pcr >= 0.9:
                bias = "⚪ NEUTRAL"
                action = "Wait for clear direction"
            elif pcr >= 0.7:
                bias = "🔴 Slightly Bearish"
                action = "Cautious approach"
            else:
                bias = "🔴 BEARISH"
                action = "Look for selling opportunities"
            
            patterns = analysis.get('patterns', [])
            pattern_summary = ""
            if patterns:
                recent = patterns[-3:]
                pattern_summary = "\n\n📍 Recent Patterns:\n"
                for p in recent:
                    pattern_summary += f"  • {p['pattern']}\n"
            
            sr_levels = analysis.get('sr_levels', {'support': [], 'resistance': []})
            sr_summary = "\n\n📊 Key Levels:\n"
            
            if sr_levels['support']:
                sr_summary += f"🟢 Support: {', '.join([f'₹{s:.0f}' for s in sr_levels['support']])}\n"
            if sr_levels['resistance']:
                sr_summary += f"🔴 Resistance: {', '.join([f'₹{r:.0f}' for r in sr_levels['resistance']])}\n"
            
            caption = f"""📊 {symbol} Option Chain Analysis

💰 Spot: ₹{analysis['current_price']:,.2f}
📅 Expiry: {analysis['expiry']}

📈 Total CE OI: {total_ce_oi:,.0f}
📉 Total PE OI: {total_pe_oi:,.0f}
🔄 Overall PCR: {pcr:.3f}

{bias}
💡 {action}{pattern_summary}{sr_summary}

⏰ {datetime.now(IST).strftime('%d-%b %H:%M IST')}

✅ Enhanced: S/R + Patterns + PCR"""
            
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=chart_buffer,
                caption=caption
            )
            
            logger.info(f"✅ Alert sent for {symbol}")
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
        except Exception as e:
            logger.error(f"❌ Error sending alert: {e}")

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
            logger.info(f"⏸️ Market closed | Current time: {now_time.strftime('%H:%M:%S IST')}")
            return
        
        logger.info("\n" + "="*60)
        logger.info(f"🔍 ANALYSIS CYCLE - {datetime.now(IST).strftime('%H:%M:%S')}")
        logger.info("="*60)
        
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
                logger.error(f"❌ Error processing {symbol}: {e}\n")
        
        logger.info("="*60)
        logger.info("✅ CYCLE COMPLETE")
        logger.info("="*60 + "\n")
    
    async def run(self):
        current_time = datetime.now(IST)
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        
        print("\n" + "="*60)
        print("🚀 ENHANCED UPSTOX BOT v3.0", flush=True)
        print("="*60)
        print(f"📅 Date: {current_time.strftime('%d-%b-%Y %A')}", flush=True)
        print(f"🕐 Time: {current_time.strftime('%H:%M:%S IST')}", flush=True)
        print(f"📊 Market: {market_status}", flush=True)
        print("✅ NEW: Support/Resistance Detection", flush=True)
        print("✅ NEW: Candlestick Pattern Recognition", flush=True)
        print("✅ NEW: Larger Chart Size (20x14)", flush=True)
        print("="*60 + "\n", flush=True)
        
        await self.client.create_session()
        
        try:
            while True:
                try:
                    await self.process_symbols()
                    
                    next_run = datetime.now(IST) + timedelta(seconds=ANALYSIS_INTERVAL)
                    logger.info(f"⏰ Next run: {next_run.strftime('%H:%M:%S')}\n")
                    
                    await asyncio.sleep(ANALYSIS_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
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
