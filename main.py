"""
🚀 MIDCPNIFTY OPTIONS BOT - v6.0 PRO (Phase 1 + Phase 2)
==========================================================
Version: 6.0 PRO (3-MIN INTERVAL + PSYCHOLOGICAL + ABSORPTION)
Author: Built for MIDCPNIFTY Options Trading
Last Updated: Feb 2026

✅ PHASE 1 FEATURES:
- 🔥 Auto Expiry Selection (Holiday-aware)
- ✅ NIFTY 50 Removed (MIDCPNIFTY standalone)
- ✅ ATM ±5 Range (125 points coverage)
- ✅ Strict Thresholds (15%, 20% for low liquidity)
- ✅ 3-Minute Analysis Interval (9:16, 9:19, 9:22...)

✅ PHASE 2 FEATURES:
- 🎯 Psychological Level Detection (12500, 13000, 13500...)
- 🔥 Absorption Logic (High Volume + Low Movement)
- ✅ Confluence Check (OI + Chart S/R Match)
- ✅ False Breakout Detection
- ✅ Enhanced Wait Signals

⚡ SPECIFICATIONS:
- Symbol: MIDCPNIFTY
- Strike Interval: 25 points
- Lot Size: 120 (as of Oct 2025)
- Expiry: Last Monday of month
- Analysis: Every 3 minutes
- DeepSeek Model: V3.2 (deepseek-reasoner)

🎯 STRATEGY:
- Primary: OI + Volume + PCR (15-min changes)
- Secondary: Psychological Levels + Absorption
- Tertiary: Confluence (OI + Chart S/R)
- AI: DeepSeek V3.2 with 30-sec timeout
- Confirmation: Triple alignment required
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import logging
import os
import pytz

# ======================== CONFIGURATION ========================
# Environment Variables
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY")

# Upstox API
UPSTOX_API_URL = "https://api.upstox.com/v2"

# Trading Parameters - MIDCPNIFTY SPECIFIC
SYMBOL = "MIDCPNIFTY"
STRIKE_INTERVAL = 25  # 25-point strikes
ATM_RANGE = 5  # ±5 strikes (125 points total coverage)
ANALYSIS_INTERVAL = 3 * 60  # 3 minutes (9:16, 9:19, 9:22...)
CACHE_SIZE = 10  # 30 min = 10 snapshots @ 3min

# Signal Thresholds - STRICT (for low liquidity)
MIN_OI_CHANGE_15MIN = 15.0  # 15% = strong signal (was 10%)
STRONG_OI_CHANGE = 20.0     # 20% = very strong (was 15%)
MIN_VOLUME_CHANGE = 20.0    # 20% volume increase (was 15%)
MIN_CONFIDENCE = 7.5        # 7.5 = very strict (was 7)

# Liquidity Filters
MIN_TOTAL_OI_FOR_SIGNAL = 100000  # 1 lakh combined OI minimum
MIN_CE_OI = 50000  # 50K minimum CE OI
MIN_PE_OI = 50000  # 50K minimum PE OI
MIN_STRIKE_VOLUME = 10000  # 10K minimum volume per strike

# Strike Weight Multipliers (adjusted for 25-point intervals)
ATM_WEIGHT = 3.0        # ATM strike gets 3x importance
NEAR_ATM_WEIGHT = 2.5   # ATM ±25 gets 2.5x importance
MID_ATM_WEIGHT = 2.0    # ATM ±50 gets 2x importance
FAR_WEIGHT = 1.0        # ATM ±75/100/125 gets 1x importance

# Psychological & Absorption Settings
PSYCHOLOGICAL_INTERVAL = 500  # Every 500 points (12500, 13000, 13500...)
ABSORPTION_VOLUME_MULTIPLIER = 2.0  # 2x average volume
ABSORPTION_PRICE_THRESHOLD = 0.5  # 0.5 ATR movement
CONFLUENCE_TOLERANCE = 50  # Within 50 points for confluence

# API Settings
API_DELAY = 0.2  # 200ms between calls
MAX_RETRIES = 3
DEEPSEEK_TIMEOUT = 30  # 30 seconds timeout

# Market Hours (IST)
IST = pytz.timezone('Asia/Kolkata')
MARKET_START_HOUR = 9
MARKET_START_MIN = 15
MARKET_END_HOUR = 15
MARKET_END_MIN = 30

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ======================== DATA STRUCTURES ========================
@dataclass
class OISnapshot:
    """Enhanced OI + Volume snapshot per strike"""
    strike: int
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_ltp: float
    pe_ltp: float
    pcr: float  # PE OI / CE OI
    timestamp: datetime


@dataclass
class MarketSnapshot:
    """Complete market data at a point in time"""
    timestamp: datetime
    spot_price: float
    atm_strike: int
    strikes_oi: Dict[int, OISnapshot]
    overall_pcr: float


@dataclass
class StrikeAnalysis:
    """Detailed analysis for a single strike with Volume + Psychological"""
    strike: int
    is_atm: bool
    distance_from_atm: int
    weight: float
    is_psychological_level: bool  # ✅ NEW
    
    # Current OI + Volume
    ce_oi: int
    pe_oi: int
    ce_volume: int
    pe_volume: int
    ce_ltp: float
    pe_ltp: float
    
    # OI Changes
    ce_oi_change_5min: float
    pe_oi_change_5min: float
    ce_oi_change_15min: float
    pe_oi_change_15min: float
    ce_oi_change_30min: float
    pe_oi_change_30min: float
    
    # Volume Changes
    ce_vol_change_5min: float
    pe_vol_change_5min: float
    ce_vol_change_15min: float
    pe_vol_change_15min: float
    ce_vol_change_30min: float
    pe_vol_change_30min: float
    
    # Ratios
    put_call_ratio: float
    pcr_change_15min: float
    
    # Writer Activity
    ce_writer_action: str
    pe_writer_action: str
    
    # Volume Confirmation
    volume_confirms_oi: bool
    volume_strength: str
    
    # Support/Resistance Role
    is_support_level: bool
    is_resistance_level: bool
    
    # Signal Strength
    bullish_signal_strength: float
    bearish_signal_strength: float
    
    # Recommendation
    strike_recommendation: str
    confidence: float


@dataclass
class SupportResistance:
    """Support/Resistance levels from OI"""
    support_strike: int
    support_put_oi: int
    resistance_strike: int
    resistance_call_oi: int
    spot_near_support: bool
    spot_near_resistance: bool


@dataclass
class ConfluenceAnalysis:
    """✅ NEW: Confluence check between OI and Chart levels"""
    support_confluence: bool
    resistance_confluence: bool
    strength: str  # "STRONG" / "MODERATE" / "WEAK"
    description: str


@dataclass
class AbsorptionSignal:
    """✅ NEW: Absorption detection"""
    is_absorbing: bool
    volume_ratio: float  # Current volume / Avg volume
    price_movement: float  # As % of ATR
    description: str


# ======================== IN-MEMORY CACHE ========================
class SimpleCache:
    """Stores last 30 min of data (10 snapshots @ 3min)"""
    
    def __init__(self, max_size: int = CACHE_SIZE):
        self.snapshots = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def add(self, snapshot: MarketSnapshot):
        """Add new snapshot"""
        async with self._lock:
            self.snapshots.append(snapshot)
            logger.info(f"📦 Cached snapshot | Total: {len(self.snapshots)} | PCR: {snapshot.overall_pcr:.2f}")
    
    async def get_minutes_ago(self, minutes: int) -> Optional[MarketSnapshot]:
        """Get snapshot from N minutes ago"""
        async with self._lock:
            if len(self.snapshots) < 2:
                return None
            
            target_time = datetime.now(IST) - timedelta(minutes=minutes)
            
            best = None
            min_diff = float('inf')
            
            for snap in self.snapshots:
                diff = abs((snap.timestamp - target_time).total_seconds())
                if diff < min_diff:
                    min_diff = diff
                    best = snap
            
            # Accept if within 90 seconds tolerance
            if best and min_diff <= 90:
                return best
            
            return None
    
    def size(self) -> int:
        return len(self.snapshots)


# ======================== UPSTOX CLIENT ========================
class UpstoxClient:
    """Upstox v2 API client for MIDCPNIFTY"""
    
    def __init__(self, token: str):
        self.token = token
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
    
    async def init(self):
        """Initialize session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=timeout
            )
    
    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """Request with retry"""
        for attempt in range(MAX_RETRIES):
            try:
                async with getattr(self.session, method)(url, **kwargs) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    elif resp.status == 429:
                        wait = (attempt + 1) * 2
                        logger.warning(f"⚠️ Rate limited, waiting {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        text = await resp.text()
                        logger.warning(f"⚠️ Request failed: {resp.status} - {text[:200]}")
                        return None
            except Exception as e:
                logger.error(f"❌ Request error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None
                await asyncio.sleep(1)
        return None
    
    async def get_option_chain(self, expiry: str) -> Optional[Dict]:
        """Get option chain for MIDCPNIFTY"""
        url = f"{UPSTOX_API_URL}/option/chain"
        params = {
            "instrument_key": "NSE_INDEX|Nifty Midcap Select",  # ✅ MIDCPNIFTY
            "expiry_date": expiry
        }
        return await self._request('get', url, params=params)
    
    async def get_1min_candles(self) -> pd.DataFrame:
        """Get MIDCPNIFTY spot 1-min candles"""
        instrument_key = "NSE_INDEX|Nifty Midcap Select"  # ✅ MIDCPNIFTY
        url = f"{UPSTOX_API_URL}/historical-candle/intraday/{instrument_key}/1minute"
        
        logger.info(f"📈 Fetching MIDCPNIFTY spot candles...")
        data = await self._request('get', url)
        
        if not data or data.get("status") != "success":
            logger.warning("⚠️ Could not fetch candle data from API")
            return pd.DataFrame()
        
        candles = data.get("data", {}).get("candles", [])
        
        if not candles or len(candles) == 0:
            logger.warning("⚠️ Empty candle data from Upstox")
            return pd.DataFrame()
        
        df_data = []
        for candle in candles:
            try:
                df_data.append({
                    'timestamp': pd.to_datetime(candle[0]),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': int(candle[5]) if len(candle) > 5 else 0
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"⚠️ Skipping malformed candle: {e}")
                continue
        
        if not df_data:
            logger.warning("⚠️ No valid candle data after parsing")
            return pd.DataFrame()
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"✅ Fetched {len(df)} 1-min MIDCPNIFTY spot candles")
        return df
    
    async def get_available_expiries(self) -> List[str]:
        """✅ UPDATED: Get all available expiry dates from Upstox API"""
        url = f"{UPSTOX_API_URL}/option/contract"
        params = {"instrument_key": "NSE_INDEX|Nifty Midcap Select"}  # ✅ MIDCPNIFTY
        
        data = await self._request('get', url, params=params)
        
        if not data or data.get("status") != "success":
            logger.warning("⚠️ Could not fetch available expiries")
            return []
        
        contracts = data.get("data", [])
        
        if not contracts:
            logger.warning("⚠️ No option contracts available")
            return []
        
        expiries = sorted(set(item.get("expiry") for item in contracts if item.get("expiry")))
        logger.info(f"📅 Found {len(expiries)} available expiries: {expiries[:3]}...")
        return expiries
    
    async def get_nearest_expiry(self) -> Optional[str]:
        """
        ✅ SMART AUTO EXPIRY SELECTION with Holiday Handling
        For MIDCPNIFTY: Last Monday of month
        """
        logger.info("🔍 Auto-detecting nearest MIDCPNIFTY expiry...")
        
        expiries = await self.get_available_expiries()
        
        if not expiries:
            logger.error("❌ No expiries available from Upstox")
            return None
        
        now = datetime.now(IST).date()
        
        # Filter future expiries (including today)
        future_expiries = [
            exp for exp in expiries 
            if datetime.strptime(exp, '%Y-%m-%d').date() >= now
        ]
        
        if not future_expiries:
            logger.warning("⚠️ No future expiries found, using last available")
            return expiries[-1]
        
        nearest = future_expiries[0]
        expiry_date = datetime.strptime(nearest, '%Y-%m-%d')
        
        # Verify it's a Monday (MIDCPNIFTY expiry day)
        if expiry_date.weekday() != 0:  # 0 = Monday
            logger.warning(f"⚠️ Expiry {nearest} is not a Monday! (Day: {expiry_date.strftime('%A')})")
            logger.info("💡 Upstox API should handle this, but flagging for awareness")
        
        logger.info(f"✅ Auto-selected nearest expiry: {nearest} ({expiry_date.strftime('%A, %d %b %Y')})")
        return nearest


# ======================== PATTERN DETECTOR ========================
class PatternDetector:
    """Enhanced candlestick pattern detection with ATR"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> List[Dict]:
        """Detect last 5 strong patterns"""
        patterns = []
        
        if df.empty or len(df) < 2:
            return patterns
        
        for i in range(len(df)):
            if i < 1:
                continue
            
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            
            body_curr = abs(curr['close'] - curr['open'])
            body_prev = abs(prev['close'] - prev['open'])
            range_curr = curr['high'] - curr['low']
            
            if range_curr == 0:
                continue
            
            # Bullish Engulfing
            if (curr['close'] > curr['open'] and 
                prev['close'] < prev['open'] and
                curr['open'] <= prev['close'] and
                curr['close'] >= prev['open'] and
                body_curr > body_prev * 1.2):
                patterns.append({
                    'time': curr.name,
                    'pattern': 'BULLISH_ENGULFING',
                    'type': 'BULLISH',
                    'strength': 8,
                    'price': curr['close']
                })
            
            # Bearish Engulfing
            elif (curr['close'] < curr['open'] and 
                  prev['close'] > prev['open'] and
                  curr['open'] >= prev['close'] and
                  curr['close'] <= prev['open'] and
                  body_curr > body_prev * 1.2):
                patterns.append({
                    'time': curr.name,
                    'pattern': 'BEARISH_ENGULFING',
                    'type': 'BEARISH',
                    'strength': 8,
                    'price': curr['close']
                })
            
            else:
                lower_wick = min(curr['open'], curr['close']) - curr['low']
                upper_wick = curr['high'] - max(curr['open'], curr['close'])
                
                # Hammer
                if (lower_wick > body_curr * 2 and 
                    upper_wick < body_curr * 0.3 and
                    body_curr < range_curr * 0.35):
                    patterns.append({
                        'time': curr.name,
                        'pattern': 'HAMMER',
                        'type': 'BULLISH',
                        'strength': 6,
                        'price': curr['close']
                    })
                
                # Shooting Star
                elif (upper_wick > body_curr * 2 and 
                      lower_wick < body_curr * 0.3 and
                      body_curr < range_curr * 0.35):
                    patterns.append({
                        'time': curr.name,
                        'pattern': 'SHOOTING_STAR',
                        'type': 'BEARISH',
                        'strength': 6,
                        'price': curr['close']
                    })
                
                # Doji
                elif body_curr < range_curr * 0.1:
                    patterns.append({
                        'time': curr.name,
                        'pattern': 'DOJI',
                        'type': 'NEUTRAL',
                        'strength': 4,
                        'price': curr['close']
                    })
        
        return patterns[-5:] if patterns else []
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate S/R from recent price action"""
        if df.empty or len(df) < 10:
            return 0.0, 0.0
        
        last_20 = df.tail(20)
        support = last_20['low'].min()
        resistance = last_20['high'].max()
        
        return support, resistance
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """✅ NEW: Calculate Average True Range"""
        if df.empty or len(df) < period:
            return 0.0
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if not atr.empty else 0.0


# ======================== PHASE 2: PSYCHOLOGICAL & ABSORPTION ========================
class PsychologicalAnalyzer:
    """✅ NEW: Psychological level detection"""
    
    @staticmethod
    def is_psychological_level(strike: int) -> bool:
        """
        Detect psychological round numbers
        MIDCPNIFTY: 12500, 13000, 13500, 14000, etc. (every 500 points)
        """
        return strike % PSYCHOLOGICAL_INTERVAL == 0
    
    @staticmethod
    def get_psychological_description(strike: int) -> str:
        """Get description for psychological level"""
        if PsychologicalAnalyzer.is_psychological_level(strike):
            return f"🎯 PSYCHOLOGICAL LEVEL: {strike} (Round number - high trader attention)"
        return ""


class AbsorptionDetector:
    """✅ NEW: Absorption logic - High volume but low price movement"""
    
    @staticmethod
    def detect(candles_df: pd.DataFrame, lookback: int = 5) -> AbsorptionSignal:
        """
        Detect absorption:
        - High volume (2x average)
        - Low price movement (< 0.5 ATR)
        """
        if candles_df.empty or len(candles_df) < lookback + 14:
            return AbsorptionSignal(
                is_absorbing=False,
                volume_ratio=0.0,
                price_movement=0.0,
                description="Not enough data for absorption check"
            )
        
        recent = candles_df.tail(lookback)
        all_data = candles_df.tail(20)  # For ATR calculation
        
        # Volume analysis
        avg_volume = all_data['volume'].mean()
        last_volume = recent['volume'].iloc[-1]
        volume_ratio = last_volume / avg_volume if avg_volume > 0 else 0
        
        # Price movement analysis
        price_range = recent['high'].max() - recent['low'].min()
        atr = PatternDetector.calculate_atr(all_data)
        price_movement_ratio = price_range / atr if atr > 0 else 0
        
        # Absorption criteria
        high_volume = volume_ratio > ABSORPTION_VOLUME_MULTIPLIER
        low_movement = price_movement_ratio < ABSORPTION_PRICE_THRESHOLD
        
        is_absorbing = high_volume and low_movement
        
        if is_absorbing:
            description = (
                f"🔥 ABSORPTION DETECTED: Volume {volume_ratio:.1f}x avg, "
                f"but price movement only {price_movement_ratio:.1f}x ATR. "
                f"Big fight happening at this level!"
            )
        else:
            description = f"Normal market: Vol {volume_ratio:.1f}x, Movement {price_movement_ratio:.1f}x ATR"
        
        return AbsorptionSignal(
            is_absorbing=is_absorbing,
            volume_ratio=volume_ratio,
            price_movement=price_movement_ratio,
            description=description
        )


class ConfluenceChecker:
    """✅ NEW: Check confluence between OI levels and Chart S/R"""
    
    @staticmethod
    def check(oi_support: int, oi_resistance: int, 
              chart_support: float, chart_resistance: float) -> ConfluenceAnalysis:
        """
        Check if OI-based levels match chart-based levels
        Tolerance: Within 50 points (CONFLUENCE_TOLERANCE)
        """
        support_diff = abs(oi_support - chart_support)
        resistance_diff = abs(oi_resistance - chart_resistance)
        
        support_match = support_diff <= CONFLUENCE_TOLERANCE
        resistance_match = resistance_diff <= CONFLUENCE_TOLERANCE
        
        # Determine strength
        if support_match and resistance_match:
            strength = "STRONG"
            description = (
                f"🔥 STRONG CONFLUENCE: Both OI Support ({oi_support}) and "
                f"OI Resistance ({oi_resistance}) match chart levels "
                f"(S: {chart_support:.0f}, R: {chart_resistance:.0f}). "
                f"These levels are CONCRETE WALLS!"
            )
        elif support_match:
            strength = "MODERATE"
            description = (
                f"⚡ MODERATE CONFLUENCE: OI Support ({oi_support}) matches "
                f"chart support ({chart_support:.0f}). "
                f"Support level is strong. "
                f"Resistance mismatch: OI {oi_resistance} vs Chart {chart_resistance:.0f}"
            )
        elif resistance_match:
            strength = "MODERATE"
            description = (
                f"⚡ MODERATE CONFLUENCE: OI Resistance ({oi_resistance}) matches "
                f"chart resistance ({chart_resistance:.0f}). "
                f"Resistance level is strong. "
                f"Support mismatch: OI {oi_support} vs Chart {chart_support:.0f}"
            )
        else:
            strength = "WEAK"
            description = (
                f"⚠️ WEAK CONFLUENCE: OI levels don't match chart levels. "
                f"OI S/R: {oi_support}/{oi_resistance}, "
                f"Chart S/R: {chart_support:.0f}/{chart_resistance:.0f}. "
                f"Be cautious of false signals."
            )
        
        return ConfluenceAnalysis(
            support_confluence=support_match,
            resistance_confluence=resistance_match,
            strength=strength,
            description=description
        )


# ======================== ENHANCED OI + VOLUME ANALYZER ========================
class EnhancedOIAnalyzer:
    """Strike-wise OI + Volume analysis with PCR + Psychological + Liquidity filters"""
    
    def __init__(self, cache: SimpleCache):
        self.cache = cache
    
    def _calculate_strike_weight(self, strike: int, atm: int) -> float:
        """Calculate weight based on distance from ATM (25-point intervals)"""
        distance = abs(strike - atm)
        
        if distance == 0:
            return ATM_WEIGHT  # 3.0
        elif distance == 25:
            return NEAR_ATM_WEIGHT  # 2.5
        elif distance == 50:
            return MID_ATM_WEIGHT  # 2.0
        else:
            return FAR_WEIGHT  # 1.0
    
    def _determine_writer_action(self, oi_change: float) -> str:
        """Determine if writers are building or unwinding"""
        if oi_change >= 15:  # Stricter threshold
            return "BUILDING"
        elif oi_change <= -15:
            return "UNWINDING"
        else:
            return "NEUTRAL"
    
    def _check_volume_confirmation(self, 
                                   oi_change: float, 
                                   vol_change: float) -> Tuple[bool, str]:
        """Check if volume confirms OI direction"""
        if oi_change > 15 and vol_change > MIN_VOLUME_CHANGE:
            return True, "STRONG"
        elif oi_change > 10 and vol_change > 15:
            return True, "MODERATE"
        elif oi_change < -15 and vol_change < -15:
            return True, "STRONG"
        elif abs(oi_change) < 10 and abs(vol_change) < 10:
            return True, "WEAK"
        else:
            return False, "WEAK"
    
    def _check_liquidity_filters(self, ce_oi: int, pe_oi: int, 
                                 ce_vol: int, pe_vol: int) -> Tuple[bool, str]:
        """✅ NEW: Check if strike meets liquidity requirements"""
        total_oi = ce_oi + pe_oi
        total_vol = ce_vol + pe_vol
        
        if total_oi < MIN_TOTAL_OI_FOR_SIGNAL:
            return False, f"Low OI: {total_oi:,} < {MIN_TOTAL_OI_FOR_SIGNAL:,}"
        
        if ce_oi < MIN_CE_OI:
            return False, f"Low CE OI: {ce_oi:,} < {MIN_CE_OI:,}"
        
        if pe_oi < MIN_PE_OI:
            return False, f"Low PE OI: {pe_oi:,} < {MIN_PE_OI:,}"
        
        if total_vol < MIN_STRIKE_VOLUME:
            return False, f"Low Volume: {total_vol:,} < {MIN_STRIKE_VOLUME:,}"
        
        return True, "Liquidity OK"
    
    def _calculate_signal_strength(self, 
                                   ce_oi_change: float, 
                                   pe_oi_change: float,
                                   ce_vol_change: float,
                                   pe_vol_change: float,
                                   weight: float,
                                   is_psychological: bool) -> Tuple[float, float]:
        """Calculate signal strength with Volume + Psychological boost"""
        
        bullish_strength = 0.0
        bearish_strength = 0.0
        
        # Volume confirmation
        ce_vol_confirms, ce_vol_strength = self._check_volume_confirmation(ce_oi_change, ce_vol_change)
        pe_vol_confirms, pe_vol_strength = self._check_volume_confirmation(pe_oi_change, pe_vol_change)
        
        # Volume multiplier
        vol_multiplier = 1.0
        if ce_vol_strength == "STRONG" or pe_vol_strength == "STRONG":
            vol_multiplier = 1.5
        elif ce_vol_strength == "WEAK" or pe_vol_strength == "WEAK":
            vol_multiplier = 0.5
        
        # Psychological level boost
        psych_multiplier = 1.2 if is_psychological else 1.0
        
        # PUT OI building = BULLISH
        if pe_oi_change >= STRONG_OI_CHANGE and pe_vol_confirms:
            bullish_strength = 9.0 * weight * vol_multiplier * psych_multiplier
        elif pe_oi_change >= MIN_OI_CHANGE_15MIN and pe_vol_confirms:
            bullish_strength = 7.0 * weight * vol_multiplier * psych_multiplier
        elif pe_oi_change >= 8:
            bullish_strength = 4.0 * weight * vol_multiplier * psych_multiplier
        
        # CALL OI building = BEARISH
        if ce_oi_change >= STRONG_OI_CHANGE and ce_vol_confirms:
            bearish_strength = 9.0 * weight * vol_multiplier * psych_multiplier
        elif ce_oi_change >= MIN_OI_CHANGE_15MIN and ce_vol_confirms:
            bearish_strength = 7.0 * weight * vol_multiplier * psych_multiplier
        elif ce_oi_change >= 8:
            bearish_strength = 4.0 * weight * vol_multiplier * psych_multiplier
        
        # PUT OI unwinding = BEARISH
        if pe_oi_change <= -STRONG_OI_CHANGE:
            bearish_strength = max(bearish_strength, 8.0 * weight * vol_multiplier * psych_multiplier)
        elif pe_oi_change <= -MIN_OI_CHANGE_15MIN:
            bearish_strength = max(bearish_strength, 6.0 * weight * vol_multiplier * psych_multiplier)
        
        # CALL OI unwinding = BULLISH
        if ce_oi_change <= -STRONG_OI_CHANGE:
            bullish_strength = max(bullish_strength, 8.0 * weight * vol_multiplier * psych_multiplier)
        elif ce_oi_change <= -MIN_OI_CHANGE_15MIN:
            bullish_strength = max(bullish_strength, 6.0 * weight * vol_multiplier * psych_multiplier)
        
        return bullish_strength, bearish_strength
    
    async def analyze_strike(self, 
                           strike: int,
                           current: MarketSnapshot,
                           snap_5min: Optional[MarketSnapshot],
                           snap_15min: Optional[MarketSnapshot],
                           snap_30min: Optional[MarketSnapshot]) -> Optional[StrikeAnalysis]:
        """Enhanced strike analysis with Volume + Psychological + Liquidity"""
        
        curr_oi = current.strikes_oi.get(strike)
        if not curr_oi:
            return None
        
        # ✅ Check liquidity filters first
        liquidity_ok, liquidity_msg = self._check_liquidity_filters(
            curr_oi.ce_oi, curr_oi.pe_oi,
            curr_oi.ce_volume, curr_oi.pe_volume
        )
        
        if not liquidity_ok:
            logger.debug(f"⚠️ Strike {strike} filtered out: {liquidity_msg}")
            # Still return analysis but with low confidence
        
        # Calculate changes
        def calc_change(current, previous):
            if previous and previous > 0:
                return ((current - previous) / previous * 100)
            return 0
        
        prev_5 = snap_5min.strikes_oi.get(strike) if snap_5min else None
        prev_15 = snap_15min.strikes_oi.get(strike) if snap_15min else None
        prev_30 = snap_30min.strikes_oi.get(strike) if snap_30min else None
        
        # OI Changes
        ce_oi_5min = calc_change(curr_oi.ce_oi, prev_5.ce_oi if prev_5 else 0)
        pe_oi_5min = calc_change(curr_oi.pe_oi, prev_5.pe_oi if prev_5 else 0)
        ce_oi_15min = calc_change(curr_oi.ce_oi, prev_15.ce_oi if prev_15 else 0)
        pe_oi_15min = calc_change(curr_oi.pe_oi, prev_15.pe_oi if prev_15 else 0)
        ce_oi_30min = calc_change(curr_oi.ce_oi, prev_30.ce_oi if prev_30 else 0)
        pe_oi_30min = calc_change(curr_oi.pe_oi, prev_30.pe_oi if prev_30 else 0)
        
        # Volume Changes
        ce_vol_5min = calc_change(curr_oi.ce_volume, prev_5.ce_volume if prev_5 else 0)
        pe_vol_5min = calc_change(curr_oi.pe_volume, prev_5.pe_volume if prev_5 else 0)
        ce_vol_15min = calc_change(curr_oi.ce_volume, prev_15.ce_volume if prev_15 else 0)
        pe_vol_15min = calc_change(curr_oi.pe_volume, prev_15.pe_volume if prev_15 else 0)
        ce_vol_30min = calc_change(curr_oi.ce_volume, prev_30.ce_volume if prev_30 else 0)
        pe_vol_30min = calc_change(curr_oi.pe_volume, prev_30.pe_volume if prev_30 else 0)
        
        # PCR change
        prev_15_pcr = prev_15.pcr if prev_15 else curr_oi.pcr
        pcr_change_15min = calc_change(curr_oi.pcr, prev_15_pcr)
        
        # Calculate weight & psychological
        is_atm = (strike == current.atm_strike)
        distance = abs(strike - current.atm_strike)
        weight = self._calculate_strike_weight(strike, current.atm_strike)
        is_psychological = PsychologicalAnalyzer.is_psychological_level(strike)
        
        # Writer actions
        ce_action = self._determine_writer_action(ce_oi_15min)
        pe_action = self._determine_writer_action(pe_oi_15min)
        
        # Volume confirmation
        vol_confirms, vol_strength = self._check_volume_confirmation(
            (ce_oi_15min + pe_oi_15min) / 2,
            (ce_vol_15min + pe_vol_15min) / 2
        )
        
        # Signal strengths (with psychological boost)
        bull_strength, bear_strength = self._calculate_signal_strength(
            ce_oi_15min, pe_oi_15min,
            ce_vol_15min, pe_vol_15min,
            weight, is_psychological
        )
        
        # Apply liquidity penalty
        if not liquidity_ok:
            bull_strength *= 0.5
            bear_strength *= 0.5
        
        # Strike recommendation
        if bull_strength >= 7 and bull_strength > bear_strength:
            recommendation = "STRONG_CALL"
            confidence = min(10, bull_strength)
        elif bear_strength >= 7 and bear_strength > bull_strength:
            recommendation = "STRONG_PUT"
            confidence = min(10, bear_strength)
        else:
            recommendation = "WAIT"
            confidence = max(bull_strength, bear_strength)
        
        return StrikeAnalysis(
            strike=strike,
            is_atm=is_atm,
            distance_from_atm=distance,
            weight=weight,
            is_psychological_level=is_psychological,
            ce_oi=curr_oi.ce_oi,
            pe_oi=curr_oi.pe_oi,
            ce_volume=curr_oi.ce_volume,
            pe_volume=curr_oi.pe_volume,
            ce_ltp=curr_oi.ce_ltp,
            pe_ltp=curr_oi.pe_ltp,
            ce_oi_change_5min=ce_oi_5min,
            pe_oi_change_5min=pe_oi_5min,
            ce_oi_change_15min=ce_oi_15min,
            pe_oi_change_15min=pe_oi_15min,
            ce_oi_change_30min=ce_oi_30min,
            pe_oi_change_30min=pe_oi_30min,
            ce_vol_change_5min=ce_vol_5min,
            pe_vol_change_5min=pe_vol_5min,
            ce_vol_change_15min=ce_vol_15min,
            pe_vol_change_15min=pe_vol_15min,
            ce_vol_change_30min=ce_vol_30min,
            pe_vol_change_30min=pe_vol_30min,
            put_call_ratio=curr_oi.pcr,
            pcr_change_15min=pcr_change_15min,
            ce_writer_action=ce_action,
            pe_writer_action=pe_action,
            volume_confirms_oi=vol_confirms,
            volume_strength=vol_strength,
            is_support_level=False,
            is_resistance_level=False,
            bullish_signal_strength=bull_strength,
            bearish_signal_strength=bear_strength,
            strike_recommendation=recommendation,
            confidence=confidence
        )
    
    async def analyze(self, current: MarketSnapshot) -> Dict:
        """Complete market analysis with Volume + PCR + Psychological"""
        snap_5min = await self.cache.get_minutes_ago(3)  # 3-min ago
        snap_15min = await self.cache.get_minutes_ago(15)
        snap_30min = await self.cache.get_minutes_ago(30)
        
        if not snap_5min:
            return {
                "available": False, 
                "reason": "Building cache (need at least 3 min)..."
            }
        
        # Analyze each strike
        strike_analyses = []
        for strike in sorted(current.strikes_oi.keys()):
            analysis = await self.analyze_strike(strike, current, snap_5min, snap_15min, snap_30min)
            if analysis:
                strike_analyses.append(analysis)
        
        # Find Support/Resistance
        support_resistance = self._find_support_resistance(current, strike_analyses)
        
        # Mark S/R strikes
        for sa in strike_analyses:
            sa.is_support_level = (sa.strike == support_resistance.support_strike)
            sa.is_resistance_level = (sa.strike == support_resistance.resistance_strike)
        
        # Overall PCR trend
        prev_15_overall_pcr = snap_15min.overall_pcr if snap_15min else current.overall_pcr
        pcr_trend = "BULLISH" if current.overall_pcr > prev_15_overall_pcr else "BEARISH"
        pcr_change_pct = ((current.overall_pcr - prev_15_overall_pcr) / prev_15_overall_pcr * 100) if prev_15_overall_pcr > 0 else 0
        
        # Overall market signal
        total_bull = sum(sa.bullish_signal_strength for sa in strike_analyses)
        total_bear = sum(sa.bearish_signal_strength for sa in strike_analyses)
        
        if total_bull > total_bear and total_bull >= 10:
            overall_signal = "BULLISH"
        elif total_bear > total_bull and total_bear >= 10:
            overall_signal = "BEARISH"
        else:
            overall_signal = "NEUTRAL"
        
        return {
            "available": True,
            "strike_analyses": strike_analyses,
            "support_resistance": support_resistance,
            "overall_signal": overall_signal,
            "total_bullish_strength": total_bull,
            "total_bearish_strength": total_bear,
            "overall_pcr": current.overall_pcr,
            "pcr_trend": pcr_trend,
            "pcr_change_pct": pcr_change_pct,
            "has_15min": snap_15min is not None,
            "has_30min": snap_30min is not None,
            "has_strong_signal": any(sa.confidence >= MIN_CONFIDENCE for sa in strike_analyses)
        }
    
    def _find_support_resistance(self, 
                                 current: MarketSnapshot,
                                 analyses: List[StrikeAnalysis]) -> SupportResistance:
        """Find S/R levels from OI"""
        
        max_put_oi = 0
        support_strike = current.atm_strike
        
        for sa in analyses:
            if sa.pe_oi > max_put_oi:
                max_put_oi = sa.pe_oi
                support_strike = sa.strike
        
        max_call_oi = 0
        resistance_strike = current.atm_strike
        
        for sa in analyses:
            if sa.ce_oi > max_call_oi:
                max_call_oi = sa.ce_oi
                resistance_strike = sa.strike
        
        spot = current.spot_price
        near_support = abs(spot - support_strike) <= 50
        near_resistance = abs(spot - resistance_strike) <= 50
        
        return SupportResistance(
            support_strike=support_strike,
            support_put_oi=max_put_oi,
            resistance_strike=resistance_strike,
            resistance_call_oi=max_call_oi,
            spot_near_support=near_support,
            spot_near_resistance=near_resistance
        )


# ======================== DEEPSEEK CLIENT ========================
class DeepSeekClient:
    """DeepSeek V3.2 API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.model = "deepseek-reasoner"  # ✅ V3.2 Reasoning model
    
    async def analyze(self, prompt: str) -> Optional[Dict]:
        """Send prompt to DeepSeek V3.2 with 30-sec timeout"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000  # Increased for Phase 2 analysis
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=DEEPSEEK_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.base_url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data['choices'][0]['message']['content']
                        
                        # Extract JSON
                        content = content.strip()
                        if content.startswith('```json'):
                            content = content[7:]
                        if content.endswith('```'):
                            content = content[:-3]
                        content = content.strip()
                        
                        return json.loads(content)
                    else:
                        logger.error(f"❌ DeepSeek API error: {resp.status}")
                        return None
        except asyncio.TimeoutError:
            logger.error(f"❌ DeepSeek timeout (>{DEEPSEEK_TIMEOUT} seconds)")
            return None
        except Exception as e:
            logger.error(f"❌ DeepSeek error: {e}")
            return None


# ======================== ENHANCED PROMPT BUILDER ========================
class EnhancedPromptBuilder:
    """Build detailed prompts with Volume + PCR + Psychological + Absorption"""
    
    @staticmethod
    def build(
        spot: float,
        atm: int,
        oi_analysis: Dict,
        candles_5min: pd.DataFrame,
        patterns: List[Dict],
        price_support: float,
        price_resistance: float,
        absorption: AbsorptionSignal,
        confluence: ConfluenceAnalysis
    ) -> str:
        """Build comprehensive prompt with ALL Phase 2 features"""
        
        now_time = datetime.now(IST).strftime('%H:%M IST')
        
        strike_analyses = oi_analysis.get("strike_analyses", [])
        sr = oi_analysis.get("support_resistance")
        overall_pcr = oi_analysis.get("overall_pcr", 0)
        pcr_trend = oi_analysis.get("pcr_trend", "NEUTRAL")
        pcr_change_pct = oi_analysis.get("pcr_change_pct", 0)
        
        # Header
        prompt = f"""You are an expert MIDCPNIFTY options trader with deep OI + Volume + Psychological analysis skills.

🎯 MARKET STATE (MIDCPNIFTY):
═══════════════════════════════
Time: {now_time}
MIDCPNIFTY Spot: ₹{spot:,.2f}
ATM Strike: {atm}
"""
        
        # Psychological check for ATM
        if PsychologicalAnalyzer.is_psychological_level(atm):
            prompt += f"🎯 ATM {atm} is a PSYCHOLOGICAL LEVEL (Round number - extra attention!)\n"
        
        prompt += f"""
📊 OVERALL PCR (Put-Call Ratio):
═════════════════════════════════
Current PCR: {overall_pcr:.2f}
15-min Change: {pcr_change_pct:+.1f}%
Trend: {pcr_trend}
"""
        
        if overall_pcr > 1.5:
            prompt += "📊 HIGH PCR → Strong PUT base (BULLISH bias)\n"
        elif overall_pcr < 0.7:
            prompt += "📊 LOW PCR → Strong CALL base (BEARISH bias)\n"
        
        # ✅ PHASE 2: Absorption
        prompt += f"""

🔥 ABSORPTION ANALYSIS:
═══════════════════════════════
{absorption.description}
Volume Ratio: {absorption.volume_ratio:.1f}x average
Price Movement: {absorption.price_movement:.1f}x ATR
Status: {"⚠️ ABSORBING (Big fight!)" if absorption.is_absorbing else "✅ Normal"}
"""
        
        if absorption.is_absorbing:
            prompt += "\n🚨 CRITICAL: High volume but low price movement - big players fighting at this level!\n"
        
        # ✅ PHASE 2: Confluence
        prompt += f"""

🎯 CONFLUENCE CHECK (OI + Chart Levels):
═══════════════════════════════
{confluence.description}
Strength: {confluence.strength}
Support Match: {"✅ YES" if confluence.support_confluence else "❌ NO"}
Resistance Match: {"✅ YES" if confluence.resistance_confluence else "❌ NO"}
"""
        
        prompt += f"""

🟢🔴 SUPPORT/RESISTANCE (OI-Based):
═══════════════════════════════
🟢 Support: {sr.support_strike} (PUT OI: {sr.support_put_oi:,})
🔴 Resistance: {sr.resistance_strike} (CALL OI: {sr.resistance_call_oi:,})
"""
        
        if sr.spot_near_support:
            prompt += f"⚡ ALERT: Spot NEAR SUPPORT ({sr.support_strike})!\n"
        if sr.spot_near_resistance:
            prompt += f"⚡ ALERT: Spot NEAR RESISTANCE ({sr.resistance_strike})!\n"
        
        # Psychological check for S/R
        if PsychologicalAnalyzer.is_psychological_level(sr.support_strike):
            prompt += f"🎯 Support {sr.support_strike} is PSYCHOLOGICAL LEVEL!\n"
        if PsychologicalAnalyzer.is_psychological_level(sr.resistance_strike):
            prompt += f"🎯 Resistance {sr.resistance_strike} is PSYCHOLOGICAL LEVEL!\n"
        
        prompt += "\n"
        
        # Strike-wise breakdown
        prompt += "📋 STRIKE-WISE OI + VOLUME ANALYSIS (15-MIN):\n"
        prompt += "═" * 70 + "\n\n"
        
        for sa in strike_analyses:
            weight_marker = ""
            if sa.weight == ATM_WEIGHT:
                weight_marker = " ⭐⭐⭐ (ATM - 3x WEIGHT)"
            elif sa.weight == NEAR_ATM_WEIGHT:
                weight_marker = " ⭐⭐ (ATM±25 - 2.5x WEIGHT)"
            elif sa.weight == MID_ATM_WEIGHT:
                weight_marker = " ⭐ (ATM±50 - 2x WEIGHT)"
            
            sr_marker = ""
            if sa.is_support_level:
                sr_marker = " 🟢 SUPPORT LEVEL"
            elif sa.is_resistance_level:
                sr_marker = " 🔴 RESISTANCE LEVEL"
            
            psych_marker = ""
            if sa.is_psychological_level:
                psych_marker = " 🎯 PSYCHOLOGICAL"
            
            vol_marker = ""
            if sa.volume_confirms_oi:
                vol_marker = f" ✅ VOL-{sa.volume_strength}"
            else:
                vol_marker = " ❌ VOL-MISMATCH (TRAP?)"
            
            prompt += f"Strike: {sa.strike}{weight_marker}{sr_marker}{psych_marker}\n"
            prompt += f"├─ CE OI: {sa.ce_oi:,} | 15min: {sa.ce_oi_change_15min:+.1f}% ({sa.ce_writer_action})\n"
            prompt += f"├─ PE OI: {sa.pe_oi:,} | 15min: {sa.pe_oi_change_15min:+.1f}% ({sa.pe_writer_action})\n"
            prompt += f"├─ CE VOL: {sa.ce_volume:,} | 15min: {sa.ce_vol_change_15min:+.1f}%{vol_marker}\n"
            prompt += f"├─ PE VOL: {sa.pe_volume:,} | 15min: {sa.pe_vol_change_15min:+.1f}%{vol_marker}\n"
            prompt += f"├─ PCR: {sa.put_call_ratio:.2f} (15min: {sa.pcr_change_15min:+.1f}%)\n"
            prompt += f"├─ Bull Strength: {sa.bullish_signal_strength:.1f}/10\n"
            prompt += f"├─ Bear Strength: {sa.bearish_signal_strength:.1f}/10\n"
            prompt += f"└─ Signal: {sa.strike_recommendation} (Conf: {sa.confidence:.1f}/10)\n\n"
        
        # Price action
        prompt += "\n📈 PRICE ACTION (Last 1 Hour - 5min candles):\n"
        prompt += "═" * 70 + "\n\n"
        
        if not candles_5min.empty and len(candles_5min) > 0:
            last_12 = candles_5min.tail(min(12, len(candles_5min)))
            for idx, row in last_12.iterrows():
                time_str = idx.strftime('%H:%M')
                o, h, l, c = row['open'], row['high'], row['low'], row['close']
                dir_emoji = "🟢" if c > o else "🔴" if c < o else "⚪"
                delta = c - o
                prompt += f"{time_str} | {o:.0f}→{c:.0f} (Δ{delta:+.0f}) | H:{h:.0f} L:{l:.0f} {dir_emoji}\n"
            
            prompt += f"\nPrice S/R: Support ₹{price_support:.2f} | Resistance ₹{price_resistance:.2f}\n"
        else:
            prompt += "No candle data available (focus on OI + Volume)\n"
        
        # Patterns
        prompt += "\n\n🕯️ KEY CANDLESTICK PATTERNS:\n"
        prompt += "═" * 70 + "\n\n"
        
        if patterns:
            for p in patterns:
                time_str = p['time'].strftime('%H:%M')
                prompt += f"{time_str}: {p['pattern']} | {p['type']} | Strength: {p['strength']}/10 | @ ₹{p['price']:.0f}\n"
        else:
            prompt += "No significant patterns detected\n"
        
        # Enhanced instructions
        prompt += f"""

🎯 ANALYSIS INSTRUCTIONS (MIDCPNIFTY v6.0 PRO):
═══════════════════════════════════════════════

🚨 CRITICAL OI + VOLUME + PSYCHOLOGICAL LOGIC:
─────────────────────────────────────────────────
✅ CORRECT INTERPRETATION:
• CALL OI ↑ + Volume ↑ = Writers Building Resistance = BEARISH → BUY_PUT
• PUT OI ↑ + Volume ↑ = Writers Building Support = BULLISH → BUY_CALL
• CALL OI ↑ but Volume ↓ = TRAP (Weak move, ignore!)
• PUT OI ↑ but Volume ↓ = TRAP (Weak move, ignore!)
• OI ↓ + Volume ↑ = Unwinding = Reversal possible

🎯 PSYCHOLOGICAL LEVELS (Every 500 points):
────────────────────────────────────────────
• 12500, 13000, 13500, 14000 etc. = Round numbers
• Big institutions place limit orders here
• These levels are HARDER TO BREAK
• Give extra weight to signals at psychological levels
• If absorption happening at psychological level → VERY STRONG LEVEL

🔥 ABSORPTION LOGIC:
────────────────────
• High Volume (>2x avg) + Low Price Movement (<0.5 ATR) = ABSORPTION
• Absorption = Big fight between bulls and bears
• If absorption at psychological level → CONCRETE WALL
• If absorption at support → Support is VERY STRONG
• If absorption at resistance → Resistance is VERY STRONG
• Recommendation: WAIT for clear breakout if absorption detected

⚡ CONFLUENCE STRENGTH:
──────────────────────
• STRONG Confluence = OI S/R matches Chart S/R = Levels are CONCRETE
• Signals at confluence levels are MORE RELIABLE
• Breakouts from confluence levels are STRONG MOVES
• Fake breakouts are LESS LIKELY at confluence levels

📊 TRIPLE CONFIRMATION REQUIRED:
────────────────────────────────
1. ✅ OI Change (15-min) → Shows writer intent
2. ✅ Volume Confirms OI → Shows real momentum
3. ✅ Candlestick Pattern OR Psychological Level → Shows market structure

ALL 3 MUST ALIGN for STRONG signal!

🎯 FOCUS PRIORITY:
──────────────────
1. ATM Strike (3x importance) - Look here FIRST
2. Check if ATM is PSYCHOLOGICAL LEVEL (extra weight)
3. Check ABSORPTION at ATM
4. Check if Volume confirms OI at ATM
5. ATM ±25/±50 Strikes (2.5x, 2x importance)
6. Support/Resistance strikes
7. Confluence check
8. Candlestick confirmation

⚡ PCR INTERPRETATION:
──────────────────────
• PCR > 1.5 = Strong PUT base → BULLISH bias
• PCR < 0.7 = Strong CALL base → BEARISH bias
• PCR ↑ = Bulls gaining strength
• PCR ↓ = Bears gaining strength

🚨 SIGNAL DECISION (STRICT for MIDCPNIFTY):
───────────────────────────────────────────
- ATM shows STRONG signal (7.5+) + Volume confirms → BUY_CALL/BUY_PUT
- ATM signal strong BUT Volume doesn't confirm → WAIT (Possible TRAP)
- Absorption detected → WAIT for clear direction
- Psychological level + Absorption → DEFINITELY WAIT
- Confluence STRONG + Signal strong → BOOST confidence
- Volume mismatch at key strikes → HIGH RISK, prefer WAIT

🚨 MIDCPNIFTY SPECIFIC:
──────────────────────
• Lower liquidity than NIFTY 50
• Higher volatility (Beta 1.10)
• 25-point strike intervals
• Be more conservative with signals
• Minimum confidence: 7.5/10 for action
• When in doubt, choose WAIT

RESPOND IN JSON:
{{
    "signal": "BUY_CALL" | "BUY_PUT" | "WAIT",
    "primary_strike": {atm},
    "confidence": 0-10,
    "stop_loss_strike": strike_number,
    "target_strike": strike_number,
    
    "atm_analysis": {{
        "ce_oi_action": "BUILDING/UNWINDING/NEUTRAL",
        "pe_oi_action": "BUILDING/UNWINDING/NEUTRAL",
        "volume_confirms": true/false,
        "volume_strength": "STRONG/MODERATE/WEAK",
        "is_psychological": {str(PsychologicalAnalyzer.is_psychological_level(atm)).lower()},
        "atm_signal": "CALL/PUT/WAIT",
        "atm_confidence": 0-10
    }},
    
    "psychological_analysis": {{
        "atm_is_psychological": {str(PsychologicalAnalyzer.is_psychological_level(atm)).lower()},
        "support_is_psychological": {str(PsychologicalAnalyzer.is_psychological_level(sr.support_strike)).lower()},
        "resistance_is_psychological": {str(PsychologicalAnalyzer.is_psychological_level(sr.resistance_strike)).lower()},
        "psychological_impact": "How psychological levels affect signal"
    }},
    
    "absorption_analysis": {{
        "is_absorbing": {str(absorption.is_absorbing).lower()},
        "absorption_impact": "How absorption affects signal",
        "should_wait_for_absorption": true/false
    }},
    
    "confluence_analysis": {{
        "strength": "{confluence.strength}",
        "support_confluence": {str(confluence.support_confluence).lower()},
        "resistance_confluence": {str(confluence.resistance_confluence).lower()},
        "confluence_boosts_confidence": true/false
    }},
    
    "pcr_analysis": {{
        "current_pcr": {overall_pcr:.2f},
        "pcr_trend": "{pcr_trend}",
        "pcr_interpretation": "What PCR tells about market sentiment",
        "pcr_supports_signal": true/false
    }},
    
    "volume_confirmation": {{
        "atm_volume_confirms_oi": true/false,
        "trap_warning": "Any volume mismatch warnings",
        "volume_quality": "STRONG/MODERATE/WEAK"
    }},
    
    "strike_breakdown": [
        {{
            "strike": {atm},
            "recommendation": "STRONG_CALL/STRONG_PUT/WAIT",
            "volume_confirms": true/false,
            "is_psychological": true/false,
            "reason": "Why this strike"
        }}
    ],
    
    "oi_support_resistance": {{
        "oi_support": {sr.support_strike if sr else atm},
        "oi_resistance": {sr.resistance_strike if sr else atm},
        "spot_position": "NEAR_SUPPORT/NEAR_RESISTANCE/MID_RANGE",
        "sr_impact": "How S/R affects trade decision"
    }},
    
    "candlestick_confirmation": {{
        "patterns_detected": ["list"],
        "patterns_confirm_oi": true/false,
        "pattern_strength": 0-10
    }},
    
    "entry_timing": {{
        "enter_now": true/false,
        "reason": "Why now or why wait (include absorption/psychological)",
        "wait_for": "What to wait for if not entering"
    }},
    
    "risk_reward": {{
        "entry_premium_estimate": 0,
        "sl_points": 0,
        "target_points": 0,
        "rr_ratio": 0
    }},
    
    "phase2_summary": {{
        "psychological_levels_detected": 0,
        "absorption_detected": {str(absorption.is_absorbing).lower()},
        "confluence_strength": "{confluence.strength}",
        "overall_market_structure": "STRONG/MODERATE/WEAK"
    }}
}}

ONLY output valid JSON, no extra text.
"""
        
        return prompt


# ======================== TELEGRAM ALERTER ========================
class TelegramAlerter:
    """Enhanced Telegram alerts with Phase 2 features"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.session = None
    
    async def _ensure_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        if self.session:
            await self.session.close()
    
    async def send_signal(self, signal: Dict, spot: float, oi_data: Dict,
                         absorption: AbsorptionSignal, confluence: ConfluenceAnalysis):
        """Send enhanced signal with Phase 2 features"""
        
        confidence = signal.get('confidence', 0)
        signal_type = signal.get('signal', 'WAIT')
        primary_strike = signal.get('primary_strike', 0)
        
        atm_analysis = signal.get('atm_analysis', {})
        psych_analysis = signal.get('psychological_analysis', {})
        abs_analysis = signal.get('absorption_analysis', {})
        conf_analysis = signal.get('confluence_analysis', {})
        pcr_analysis = signal.get('pcr_analysis', {})
        volume_conf = signal.get('volume_confirmation', {})
        phase2_summary = signal.get('phase2_summary', {})
        
        message = f"""🚨 MIDCPNIFTY v6.0 PRO SIGNAL

⏰ {datetime.now(IST).strftime('%d-%b %H:%M:%S IST')}

💰 Spot: ₹{spot:,.2f}
📊 Signal: <b>{signal_type}</b>
⭐ Confidence: {confidence}/10

💼 TRADE SETUP:
━━━━━━━━━━━━━━━━━━━
Entry: {primary_strike} {"CE" if "CALL" in signal_type else "PE" if "PUT" in signal_type else ""}
SL: {signal.get('stop_loss_strike', 'N/A')}
Target: {signal.get('target_strike', 'N/A')}
RR: {signal.get('risk_reward', {}).get('rr_ratio', 'N/A')}

📊 ATM ANALYSIS:
━━━━━━━━━━━━━━━━━━━
CE Writers: {atm_analysis.get('ce_oi_action', 'N/A')}
PE Writers: {atm_analysis.get('pe_oi_action', 'N/A')}
Volume Confirms: {"✅" if atm_analysis.get('volume_confirms') else "❌ TRAP WARNING!"}
Psychological: {"🎯 YES" if atm_analysis.get('is_psychological') else "❌ NO"}
Signal: {atm_analysis.get('atm_signal', 'N/A')}

🎯 PSYCHOLOGICAL LEVELS:
━━━━━━━━━━━━━━━━━━━
ATM Psychological: {"✅ YES" if psych_analysis.get('atm_is_psychological') else "❌ NO"}
Support Psychological: {"✅ YES" if psych_analysis.get('support_is_psychological') else "❌ NO"}
Resistance Psychological: {"✅ YES" if psych_analysis.get('resistance_is_psychological') else "❌ NO"}

🔥 ABSORPTION CHECK:
━━━━━━━━━━━━━━━━━━━
Status: {"⚠️ ABSORBING" if abs_analysis.get('is_absorbing') else "✅ Normal"}
Should Wait: {"⏳ YES" if abs_analysis.get('should_wait_for_absorption') else "✅ NO"}
"""
        
        if absorption.is_absorbing:
            message += f"⚠️ {absorption.description}\n"
        
        message += f"""
⚡ CONFLUENCE:
━━━━━━━━━━━━━━━━━━━
Strength: {conf_analysis.get('strength', 'N/A')}
Boosts Confidence: {"✅ YES" if conf_analysis.get('confluence_boosts_confidence') else "❌ NO"}

📈 PCR ANALYSIS:
━━━━━━━━━━━━━━━━━━━
Current PCR: {pcr_analysis.get('current_pcr', 'N/A')}
Trend: {pcr_analysis.get('pcr_trend', 'N/A')}
Supports Signal: {"✅ YES" if pcr_analysis.get('pcr_supports_signal') else "❌ NO"}

⚡ VOLUME CHECK:
━━━━━━━━━━━━━━━━━━━
ATM Vol Confirms: {"✅ YES" if volume_conf.get('atm_volume_confirms_oi') else "❌ NO - CAUTION"}
Quality: {volume_conf.get('volume_quality', 'N/A')}
"""
        
        if volume_conf.get('trap_warning'):
            message += f"⚠️ WARNING: {volume_conf.get('trap_warning')}\n"
        
        message += f"""
⏰ ENTRY TIMING:
━━━━━━━━━━━━━━━━━━━
Enter Now: {"✅ YES" if signal.get('entry_timing', {}).get('enter_now') else "⏳ WAIT"}
Reason: {signal.get('entry_timing', {}).get('reason', 'N/A')}

🎯 PHASE 2 SUMMARY:
━━━━━━━━━━━━━━━━━━━
Psychological Levels: {phase2_summary.get('psychological_levels_detected', 0)}
Absorption: {"⚠️ YES" if phase2_summary.get('absorption_detected') else "✅ NO"}
Confluence: {phase2_summary.get('confluence_strength', 'N/A')}
Market Structure: {phase2_summary.get('overall_market_structure', 'N/A')}

━━━━━━━━━━━━━━━━━━━
🤖 DeepSeek V3.2 Reasoner
📊 3-Min Interval | Phase 1+2
"""
        
        try:
            await self._ensure_session()
            
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            
            async with self.session.post(url, json=payload) as resp:
                if resp.status == 200:
                    logger.info("✅ Enhanced alert sent to Telegram")
                else:
                    error_text = await resp.text()
                    logger.error(f"❌ Telegram error: {resp.status} - {error_text}")
        
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")


# ======================== MAIN BOT ========================
class MidcpNiftyBot:
    """MIDCPNIFTY v6.0 PRO Bot with Phase 1 + Phase 2"""
    
    def __init__(self):
        self.upstox = UpstoxClient(UPSTOX_ACCESS_TOKEN)
        self.cache = SimpleCache()
        self.oi_analyzer = EnhancedOIAnalyzer(self.cache)
        self.deepseek = DeepSeekClient(DEEPSEEK_API_KEY)
        self.alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.pattern_detector = PatternDetector()
        self.prompt_builder = EnhancedPromptBuilder()
        self.absorption_detector = AbsorptionDetector()
        self.confluence_checker = ConfluenceChecker()
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now(IST)
        
        if now.weekday() >= 5:
            return False
        
        market_start = now.replace(hour=MARKET_START_HOUR, minute=MARKET_START_MIN)
        market_end = now.replace(hour=MARKET_END_HOUR, minute=MARKET_END_MIN)
        
        return market_start <= now <= market_end
    
    async def fetch_market_data(self) -> Optional[MarketSnapshot]:
        """Fetch MIDCPNIFTY market data"""
        try:
            expiry = await self.upstox.get_nearest_expiry()
            if not expiry:
                logger.warning("⚠️ Could not determine expiry")
                return None
            
            logger.info(f"📅 Using expiry: {expiry}")
            
            await asyncio.sleep(API_DELAY)
            chain_data = await self.upstox.get_option_chain(expiry)
            
            if not chain_data or chain_data.get("status") != "success":
                logger.warning("⚠️ Could not fetch option chain")
                return None
            
            chain = chain_data.get("data", [])
            
            if not chain or len(chain) == 0:
                logger.warning(f"⚠️ Empty option chain")
                return None
            
            # Extract spot
            spot = 0.0
            for item in chain:
                spot = item.get("underlying_spot_price", 0.0)
                if spot > 0:
                    break
            
            if spot == 0:
                logger.warning("⚠️ Could not extract spot price")
                return None
            
            logger.info(f"💰 MIDCPNIFTY Spot: ₹{spot:,.2f}")
            
            # Calculate ATM
            atm = round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL
            
            # Extract strikes
            min_strike = atm - (ATM_RANGE * STRIKE_INTERVAL)
            max_strike = atm + (ATM_RANGE * STRIKE_INTERVAL)
            
            strikes_oi = {}
            total_ce_oi = 0
            total_pe_oi = 0
            
            for item in chain:
                strike = item.get("strike_price")
                
                if not (min_strike <= strike <= max_strike):
                    continue
                
                ce_data = item.get("call_options", {}).get("market_data", {})
                pe_data = item.get("put_options", {}).get("market_data", {})
                
                ce_oi = ce_data.get("oi", 0)
                pe_oi = pe_data.get("oi", 0)
                ce_volume = ce_data.get("volume", 0)
                pe_volume = pe_data.get("volume", 0)
                
                total_ce_oi += ce_oi
                total_pe_oi += pe_oi
                
                pcr = (pe_oi / ce_oi) if ce_oi > 0 else 0
                
                strikes_oi[strike] = OISnapshot(
                    strike=strike,
                    ce_oi=ce_oi,
                    pe_oi=pe_oi,
                    ce_volume=ce_volume,
                    pe_volume=pe_volume,
                    ce_ltp=ce_data.get("ltp", 0.0),
                    pe_ltp=pe_data.get("ltp", 0.0),
                    pcr=pcr,
                    timestamp=datetime.now(IST)
                )
            
            if not strikes_oi:
                logger.warning(f"⚠️ No strikes found")
                return None
            
            overall_pcr = (total_pe_oi / total_ce_oi) if total_ce_oi > 0 else 0
            
            logger.info(f"📊 Fetched {len(strikes_oi)} strikes | ATM: {atm} | PCR: {overall_pcr:.2f}")
            
            return MarketSnapshot(
                timestamp=datetime.now(IST),
                spot_price=spot,
                atm_strike=atm,
                strikes_oi=strikes_oi,
                overall_pcr=overall_pcr
            )
            
        except Exception as e:
            logger.error(f"❌ Error fetching data: {e}")
            logger.exception("Full traceback:")
            return None
    
    def _log_detailed_analysis_data(self, 
                                    current: MarketSnapshot,
                                    oi_analysis: Dict,
                                    candles_5min: pd.DataFrame,
                                    patterns: List[Dict],
                                    absorption: AbsorptionSignal,
                                    confluence: ConfluenceAnalysis):
        """Log all analysis data BEFORE sending to DeepSeek"""
        logger.info("\n" + "="*70)
        logger.info("📊 DETAILED ANALYSIS DATA v6.0 PRO (Phase 1+2)")
        logger.info("="*70)
        
        # Market State
        logger.info(f"\n⏰ Time: {current.timestamp.strftime('%H:%M:%S IST')}")
        logger.info(f"💰 Spot: ₹{current.spot_price:,.2f}")
        logger.info(f"📅 ATM: {current.atm_strike}")
        
        if PsychologicalAnalyzer.is_psychological_level(current.atm_strike):
            logger.info(f"🎯 ATM {current.atm_strike} is PSYCHOLOGICAL LEVEL!")
        
        logger.info(f"📈 Overall PCR: {current.overall_pcr:.2f} ({oi_analysis.get('pcr_trend', 'N/A')})")
        logger.info(f"📊 PCR Change (15min): {oi_analysis.get('pcr_change_pct', 0):+.1f}%")
        
        # Phase 2: Absorption
        logger.info(f"\n🔥 ABSORPTION STATUS:")
        logger.info(f"  {absorption.description}")
        if absorption.is_absorbing:
            logger.info("  ⚠️ BIG FIGHT DETECTED AT THIS LEVEL!")
        
        # Phase 2: Confluence
        logger.info(f"\n⚡ CONFLUENCE:")
        logger.info(f"  Strength: {confluence.strength}")
        logger.info(f"  {confluence.description[:100]}...")
        
        # S/R Levels
        sr = oi_analysis.get("support_resistance")
        if sr:
            logger.info(f"\n🟢 Support: {sr.support_strike} (PUT OI: {sr.support_put_oi:,})")
            if PsychologicalAnalyzer.is_psychological_level(sr.support_strike):
                logger.info("  🎯 Support is PSYCHOLOGICAL LEVEL!")
            
            logger.info(f"🔴 Resistance: {sr.resistance_strike} (CALL OI: {sr.resistance_call_oi:,})")
            if PsychologicalAnalyzer.is_psychological_level(sr.resistance_strike):
                logger.info("  🎯 Resistance is PSYCHOLOGICAL LEVEL!")
            
            if sr.spot_near_support:
                logger.info("⚡ Spot NEAR SUPPORT!")
            if sr.spot_near_resistance:
                logger.info("⚡ Spot NEAR RESISTANCE!")
        
        # Strike-wise
        logger.info("\n📊 STRIKE-WISE OI + VOLUME (15-min):")
        logger.info("-" * 70)
        
        strike_analyses = oi_analysis.get("strike_analyses", [])
        for sa in strike_analyses:
            atm_marker = " (ATM ⭐⭐⭐)" if sa.is_atm else ""
            psych_marker = " 🎯PSYCH" if sa.is_psychological_level else ""
            logger.info(f"\nStrike {sa.strike}{atm_marker}{psych_marker}:")
            logger.info(f"  CE OI: {sa.ce_oi:,} | 15min: {sa.ce_oi_change_15min:+.1f}% | {sa.ce_writer_action}")
            logger.info(f"  PE OI: {sa.pe_oi:,} | 15min: {sa.pe_oi_change_15min:+.1f}% | {sa.pe_writer_action}")
            logger.info(f"  CE Vol: {sa.ce_volume:,} | 15min: {sa.ce_vol_change_15min:+.1f}%")
            logger.info(f"  PE Vol: {sa.pe_volume:,} | 15min: {sa.pe_vol_change_15min:+.1f}%")
            logger.info(f"  Vol Confirms: {'✅ YES' if sa.volume_confirms_oi else '❌ NO (TRAP?)'}")
            logger.info(f"  Bull: {sa.bullish_signal_strength:.1f}/10 | Bear: {sa.bearish_signal_strength:.1f}/10")
            logger.info(f"  Recommendation: {sa.strike_recommendation} (Conf: {sa.confidence:.1f}/10)")
        
        # Candlestick
        logger.info("\n🕯️ CANDLESTICK DATA (Last 12 x 5-min):")
        logger.info("-" * 70)
        
        if not candles_5min.empty and len(candles_5min) > 0:
            last_12 = candles_5min.tail(12)
            for idx, row in last_12.iterrows():
                time_str = idx.strftime('%H:%M')
                o, h, l, c = row['open'], row['high'], row['low'], row['close']
                delta = c - o
                dir_emoji = "🟢" if delta > 0 else "🔴" if delta < 0 else "⚪"
                logger.info(f"  {time_str} | {o:.0f}→{c:.0f} (Δ{delta:+.0f}) | H:{h:.0f} L:{l:.0f} {dir_emoji}")
        else:
            logger.info("  No candle data available")
        
        # Patterns
        if patterns:
            logger.info("\n🎯 DETECTED PATTERNS:")
            logger.info("-" * 70)
            for p in patterns:
                logger.info(f"  {p['time'].strftime('%H:%M')}: {p['pattern']} | {p['type']} | {p['strength']}/10")
        
        logger.info("\n" + "="*70)
        logger.info("🤖 Now sending to DeepSeek V3.2 Reasoner...")
        logger.info("="*70 + "\n")
    
    async def analyze_cycle(self):
        """Main enhanced analysis cycle with Phase 1 + Phase 2"""
        logger.info("\n" + "="*70)
        logger.info(f"🔍 ANALYSIS CYCLE v6.0 PRO - {datetime.now(IST).strftime('%H:%M:%S')}")
        logger.info("="*70)
        
        # Fetch data
        current_snapshot = await self.fetch_market_data()
        
        if not current_snapshot:
            logger.warning("⚠️ Skipping cycle - no data")
            return
        
        # Add to cache
        await self.cache.add(current_snapshot)
        
        # OI analysis
        oi_analysis = await self.oi_analyzer.analyze(current_snapshot)
        
        if not oi_analysis.get("available"):
            logger.info(f"⏳ {oi_analysis.get('reason', 'Building cache...')}")
            return
        
        # Check for strong signals
        if not oi_analysis.get("has_strong_signal"):
            logger.info("📊 No strong signals (all < 7.5 confidence)")
            return
        
        logger.info("🚨 Strong signal detected! Proceeding...")
        
        # Fetch candles
        candles_1min = await self.upstox.get_1min_candles()
        
        # Resample to 5-min
        if not candles_1min.empty and len(candles_1min) >= 5:
            try:
                candles_5min = candles_1min.resample('5min').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                logger.info(f"📊 Resampled to {len(candles_5min)} 5-min candles")
            except Exception as e:
                logger.warning(f"⚠️ Resampling error: {e}")
                candles_5min = pd.DataFrame()
        else:
            candles_5min = pd.DataFrame()
        
        # Detect patterns
        patterns = self.pattern_detector.detect(candles_5min) if not candles_5min.empty else []
        
        # Calculate price S/R
        price_support, price_resistance = self.pattern_detector.calculate_support_resistance(candles_5min)
        
        # ✅ PHASE 2: Absorption detection
        absorption = self.absorption_detector.detect(candles_5min) if not candles_5min.empty else AbsorptionSignal(
            is_absorbing=False, volume_ratio=0, price_movement=0, description="No data"
        )
        
        # ✅ PHASE 2: Confluence check
        sr = oi_analysis.get("support_resistance")
        confluence = self.confluence_checker.check(
            sr.support_strike, sr.resistance_strike,
            price_support, price_resistance
        ) if sr else ConfluenceAnalysis(
            support_confluence=False, resistance_confluence=False,
            strength="WEAK", description="No S/R data"
        )
        
        # Log detailed analysis
        self._log_detailed_analysis_data(
            current_snapshot, oi_analysis, candles_5min, 
            patterns, absorption, confluence
        )
        
        # Build prompt
        prompt = self.prompt_builder.build(
            spot=current_snapshot.spot_price,
            atm=current_snapshot.atm_strike,
            oi_analysis=oi_analysis,
            candles_5min=candles_5min,
            patterns=patterns,
            price_support=price_support,
            price_resistance=price_resistance,
            absorption=absorption,
            confluence=confluence
        )
        
        logger.info(f"🤖 Sending to DeepSeek V3.2 Reasoner (timeout: {DEEPSEEK_TIMEOUT}s)...")
        
        # Get AI signal
        ai_signal = await self.deepseek.analyze(prompt)
        
        if not ai_signal:
            logger.warning("⚠️ DeepSeek timeout - using fallback")
            
            # Fallback logic
            strike_analyses = oi_analysis.get("strike_analyses", [])
            atm_strike = next((sa for sa in strike_analyses if sa.is_atm), None)
            
            if atm_strike and atm_strike.volume_confirms_oi and not absorption.is_absorbing:
                if atm_strike.bullish_signal_strength > atm_strike.bearish_signal_strength:
                    fallback_signal = "BUY_CALL"
                    fallback_conf = min(10, atm_strike.bullish_signal_strength)
                else:
                    fallback_signal = "BUY_PUT"
                    fallback_conf = min(10, atm_strike.bearish_signal_strength)
            else:
                fallback_signal = "WAIT"
                fallback_conf = 3
            
            ai_signal = {
                'signal': fallback_signal,
                'confidence': fallback_conf,
                'primary_strike': current_snapshot.atm_strike,
                'atm_analysis': {
                    'volume_confirms': atm_strike.volume_confirms_oi if atm_strike else False,
                    'is_psychological': PsychologicalAnalyzer.is_psychological_level(current_snapshot.atm_strike)
                },
                'absorption_analysis': {'is_absorbing': absorption.is_absorbing},
                'confluence_analysis': {'strength': confluence.strength},
                'volume_confirmation': {'trap_warning': 'AI unavailable'},
                'entry_timing': {'enter_now': False, 'reason': 'AI timeout'}
            }
        
        confidence = ai_signal.get('confidence', 0)
        signal_type = ai_signal.get('signal', 'WAIT')
        
        logger.info(f"🎯 Signal: {signal_type} | Confidence: {confidence}/10")
        
        # Send alert
        if confidence >= MIN_CONFIDENCE:
            logger.info("✅ Sending Telegram alert...")
            await self.alerter.send_signal(
                ai_signal, current_snapshot.spot_price, 
                oi_analysis, absorption, confluence
            )
        else:
            logger.info(f"⏳ Low confidence ({confidence}/10), no alert")
        
        logger.info("="*70 + "\n")
    
    async def run(self):
        """Main bot loop"""
        logger.info("\n" + "="*70)
        logger.info("🚀 MIDCPNIFTY OPTIONS BOT v6.0 PRO")
        logger.info("="*70)
        logger.info(f"📅 {datetime.now(IST).strftime('%d-%b-%Y %A')}")
        logger.info(f"🕐 {datetime.now(IST).strftime('%H:%M:%S IST')}")
        logger.info(f"⏱️  Interval: {ANALYSIS_INTERVAL // 60} minutes (3-min)")
        logger.info(f"📊 Symbol: MIDCPNIFTY")
        logger.info(f"📊 Strike Interval: 25 points")
        logger.info(f"📊 ATM Range: ±{ATM_RANGE} strikes (±{ATM_RANGE * STRIKE_INTERVAL} points)")
        logger.info(f"🎯 Features: Phase 1 + Phase 2")
        logger.info(f"  • Auto Expiry | Strict Thresholds")
        logger.info(f"  • Psychological Levels | Absorption")
        logger.info(f"  • Confluence Check | Volume Confirmation")
        logger.info(f"🤖 AI: DeepSeek V3.2 Reasoner ({DEEPSEEK_TIMEOUT}s timeout)")
        logger.info("="*70 + "\n")
        
        await self.upstox.init()
        
        try:
            while True:
                try:
                    if self.is_market_open():
                        await self.analyze_cycle()
                    else:
                        logger.info("💤 Market closed")
                    
                    next_run = datetime.now(IST) + timedelta(seconds=ANALYSIS_INTERVAL)
                    logger.info(f"⏰ Next: {next_run.strftime('%H:%M:%S')}\n")
                    
                    await asyncio.sleep(ANALYSIS_INTERVAL)
                
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}")
                    logger.exception("Traceback:")
                    await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopped")
        
        finally:
            await self.upstox.close()
            await self.alerter.close()
            logger.info("👋 Closed")


# ======================== HTTP WRAPPER ========================
async def health_check(request):
    """Health endpoint"""
    return aiohttp.web.Response(text="✅ MIDCPNIFTY v6.0 PRO Running! (3-min interval)")


async def start_bot_background(app):
    """Start bot"""
    app['bot_task'] = asyncio.create_task(run_trading_bot())


async def run_trading_bot():
    """Run bot"""
    bot = MidcpNiftyBot()
    await bot.run()


# ======================== ENTRY POINT ========================
if __name__ == "__main__":
    from aiohttp import web
    
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    app.on_startup.append(start_bot_background)
    
    port = int(os.getenv('PORT', 8001))  # Different port for MIDCPNIFTY
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║   🚀 MIDCPNIFTY OPTIONS BOT v6.0 PRO                ║
║   PHASE 1 + PHASE 2 COMPLETE!                       ║
╚══════════════════════════════════════════════════════╝

✅ PHASE 1 FEATURES:
  • 3-Minute Analysis Interval (9:16, 9:19, 9:22...)
  • Auto Expiry Selection (Holiday-aware)
  • NIFTY 50 Removed (MIDCPNIFTY standalone)
  • ATM ±5 Range (125 points coverage)
  • Strict Thresholds (15%, 20%)
  • Enhanced Liquidity Filters

✅ PHASE 2 FEATURES:
  • Psychological Level Detection (12500, 13000, 13500...)
  • Absorption Logic (High Volume + Low Movement)
  • Confluence Check (OI + Chart S/R Match)
  • False Breakout Detection
  • Enhanced Wait Signals

⚡ SPECIFICATIONS:
  • Symbol: MIDCPNIFTY
  • Strike Interval: 25 points
  • Lot Size: 120
  • Expiry: Monday (auto-detected)
  • AI Model: DeepSeek V3.2 Reasoner
  • Minimum Confidence: 7.5/10

Starting on port {port}...
Bot running in background.
""")
    
    web.run_app(app, host='0.0.0.0', port=port)
