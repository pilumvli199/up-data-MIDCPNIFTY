import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time, timedelta
import json
import logging
from typing import Dict, List, Optional
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
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# Trading params
ANALYSIS_INTERVAL = 5 * 60  # 5 minutes
CANDLES_COUNT = 50
ATM_RANGE = 5  # ±5 strikes

# Market hours (IST)
MARKET_START = dt_time(9, 15)
MARKET_END = dt_time(15, 30)
IST = pytz.timezone('Asia/Kolkata')

# NIFTY 50 stocks with ISIN codes (NSE symbols)
NIFTY50_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "BAJFINANCE", "KOTAKBANK", "LT", "ASIANPAINT", 
    "AXISBANK", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND",
    "WIPRO", "M&M", "NTPC", "HCLTECH", "TATAMOTORS", "TATASTEEL", "POWERGRID",
    "BAJAJFINSV", "ONGC", "TECHM", "ADANIENT", "COALINDIA", "JSWSTEEL",
    "INDUSINDBK", "GRASIM", "HINDALCO", "HDFCLIFE", "BRITANNIA", "CIPLA",
    "BPCL", "EICHERMOT", "DRREDDY", "APOLLOHOSP", "DIVISLAB", "SBILIFE",
    "BAJAJ-AUTO", "TATACONSUM", "HEROMOTOCO", "ADANIPORTS", "LTIM", "UPL"
]

INDICES = ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== UPSTOX API CLIENT ========================
class UpstoxClient:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
    async def create_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    def get_instrument_key(self, symbol: str, segment: str = "NSE_INDEX") -> str:
        """Get Upstox instrument key"""
        # For indices: NSE_INDEX|Nifty 50
        # For stocks: NSE_EQ|INE002A01018 (ISIN)
        # For options: NFO|NIFTY24DEC19C24000
        
        if symbol in ["NIFTY", "BANKNIFTY", "MIDCPNIFTY", "FINNIFTY"]:
            mapping = {
                "NIFTY": "NSE_INDEX|Nifty 50",
                "BANKNIFTY": "NSE_INDEX|Nifty Bank",
                "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
                "FINNIFTY": "NSE_INDEX|Nifty Fin Service"
            }
            return mapping.get(symbol, f"NSE_INDEX|{symbol}")
        else:
            # Stock - use NSE_EQ segment with symbol
            return f"NSE_EQ|{symbol}"
    
    async def get_ltp(self, instrument_key: str) -> float:
        """Get Last Traded Price"""
        try:
            url = f"{UPSTOX_API_URL}/market-quote/ltp"
            params = {"instrument_key": instrument_key}
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("status") == "success":
                    return data["data"][instrument_key]["last_price"]
                return 0.0
        except Exception as e:
            logger.error(f"❌ Error fetching LTP: {e}")
            return 0.0
    
    async def get_historical_candles(self, instrument_key: str, count: int = 50) -> pd.DataFrame:
        """Get historical intraday candles (5 min)"""
        try:
            url = f"{UPSTOX_API_URL}/historical-candle/intraday/{instrument_key}/5minute"
            
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
    
    async def get_option_chain(self, symbol: str, expiry: str) -> Dict:
        """Get option chain data"""
        try:
            # Upstox option chain endpoint
            url = f"{UPSTOX_API_URL}/option/chain"
            params = {
                "instrument_key": self.get_instrument_key(symbol),
                "expiry_date": expiry
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("status") == "success":
                    return data["data"]
                
                return {}
        except Exception as e:
            logger.error(f"❌ Error fetching option chain: {e}")
            return {}
    
    async def get_expiries(self, symbol: str) -> List[str]:
        """Get available expiries for symbol"""
        try:
            # This is a placeholder - Upstox may have different endpoint
            # For now, calculate nearest Thursday (weekly) / last Thursday (monthly)
            
            today = datetime.now(IST).date()
            
            # Find next Thursday
            days_ahead = (3 - today.weekday()) % 7  # Thursday = 3
            if days_ahead == 0:
                days_ahead = 7
            
            next_expiry = today + timedelta(days=days_ahead)
            
            return [next_expiry.strftime("%Y-%m-%d")]
            
        except Exception as e:
            logger.error(f"❌ Error calculating expiry: {e}")
            return []

# ======================== OPTION CHAIN ANALYZER ========================
class OptionAnalyzer:
    def __init__(self, client: UpstoxClient):
        self.client = client
    
    async def get_atm_strikes(self, current_price: float, symbol: str) -> List[float]:
        """Calculate ATM ±5 strikes"""
        # Determine strike interval based on symbol
        intervals = {
            "NIFTY": 50,
            "BANKNIFTY": 100,
            "MIDCPNIFTY": 25,
            "FINNIFTY": 50
        }
        
        interval = intervals.get(symbol, 100)  # Default 100 for stocks
        
        # Round to nearest strike
        atm = round(current_price / interval) * interval
        
        strikes = []
        for i in range(-ATM_RANGE, ATM_RANGE + 1):
            strikes.append(atm + (i * interval))
        
        return sorted(strikes)
    
    async def fetch_strike_data(self, symbol: str, expiry: str, strike: float, option_type: str) -> Dict:
        """Fetch data for a specific strike"""
        try:
            # Build option instrument key
            # Format: NFO|NIFTY24DEC19C24000
            exp_date = datetime.strptime(expiry, "%Y-%m-%d")
            exp_str = exp_date.strftime("%y%b%d").upper()
            
            opt_type = "C" if option_type == "CE" else "P"
            strike_str = str(int(strike))
            
            instrument_key = f"NFO|{symbol}{exp_str}{opt_type}{strike_str}"
            
            # Get LTP
            ltp = await self.client.get_ltp(instrument_key)
            
            # For OI and volume, we'd need market depth/quote endpoint
            # Placeholder for now
            oi = 0
            volume = 0
            
            return {
                "strike": strike,
                "ltp": ltp,
                "oi": oi,
                "volume": volume
            }
            
        except Exception as e:
            logger.debug(f"Error fetching {option_type} {strike}: {e}")
            return {
                "strike": strike,
                "ltp": 0,
                "oi": 0,
                "volume": 0
            }
    
    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Complete analysis for a symbol"""
        try:
            logger.info(f"📊 Analyzing {symbol}...")
            
            # Get current price
            instrument_key = self.client.get_instrument_key(symbol)
            current_price = await self.client.get_ltp(instrument_key)
            
            if current_price == 0:
                logger.warning(f"⚠️ Could not fetch price for {symbol}")
                return None
            
            logger.info(f"   💰 LTP: ₹{current_price:,.2f}")
            
            # Get candles
            candles = await self.client.get_historical_candles(instrument_key, CANDLES_COUNT)
            
            if candles.empty:
                logger.warning(f"⚠️ No candle data for {symbol}")
                return None
            
            logger.info(f"   📈 Fetched {len(candles)} candles")
            
            # Get nearest expiry
            expiries = await self.client.get_expiries(symbol)
            if not expiries:
                logger.warning(f"⚠️ No expiry found for {symbol}")
                return None
            
            expiry = expiries[0]
            logger.info(f"   📅 Expiry: {expiry}")
            
            # Get ATM strikes
            strikes = await self.get_atm_strikes(current_price, symbol)
            logger.info(f"   🎯 Strikes: {strikes[0]} to {strikes[-1]}")
            
            # Fetch option data for all strikes
            ce_data = []
            pe_data = []
            
            for strike in strikes:
                ce = await self.fetch_strike_data(symbol, expiry, strike, "CE")
                pe = await self.fetch_strike_data(symbol, expiry, strike, "PE")
                
                ce_data.append(ce)
                pe_data.append(pe)
                
                await asyncio.sleep(0.1)  # Rate limiting
            
            # Calculate PCR
            total_ce_oi = sum(d["oi"] for d in ce_data)
            total_pe_oi = sum(d["oi"] for d in pe_data)
            total_ce_vol = sum(d["volume"] for d in ce_data)
            total_pe_vol = sum(d["volume"] for d in pe_data)
            
            pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "expiry": expiry,
                "candles": candles,
                "strikes": strikes,
                "ce_data": ce_data,
                "pe_data": pe_data,
                "pcr_oi": pcr_oi,
                "pcr_vol": pcr_vol
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None

# ======================== CHART GENERATOR ========================
class ChartGenerator:
    @staticmethod
    def create_combined_chart(analysis: Dict) -> BytesIO:
        """Create combined candlestick + option chain PNG"""
        symbol = analysis["symbol"]
        candles = analysis["candles"]
        current_price = analysis["current_price"]
        
        # Create figure with GridSpec
        fig = plt.figure(figsize=(16, 12), facecolor='white')
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
        
        # ========== CANDLESTICK CHART ==========
        ax1 = fig.add_subplot(gs[0])
        
        # Custom style (TradingView white)
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
        
        # Plot candlestick
        mpf.plot(
            candles,
            type='candle',
            style=s,
            ax=ax1,
            volume=False,
            show_nontrading=False
        )
        
        ax1.set_title(f"{symbol} - 5min Chart | LTP: ₹{current_price:,.2f}", 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # ========== OPTION CHAIN TABLE ==========
        ax2 = fig.add_subplot(gs[1:])
        ax2.axis('tight')
        ax2.axis('off')
        
        # Prepare table data
        table_data = [["Strike", "CE OI", "CE Vol", "CE LTP", "PE LTP", "PE Vol", "PE OI"]]
        
        for i, strike in enumerate(analysis["strikes"]):
            ce = analysis["ce_data"][i]
            pe = analysis["pe_data"][i]
            
            row = [
                f"₹{strike:,.0f}",
                f"{ce['oi']:,}",
                f"{ce['volume']:,}",
                f"₹{ce['ltp']:.2f}",
                f"₹{pe['ltp']:.2f}",
                f"{pe['volume']:,}",
                f"{pe['oi']:,}"
            ]
            table_data.append(row)
        
        # Add PCR row
        table_data.append([
            "PCR",
            "",
            "",
            f"OI: {analysis['pcr_oi']:.2f}",
            f"Vol: {analysis['pcr_vol']:.2f}",
            "",
            ""
        ])
        
        # Create table
        table = ax2.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.12] * 7
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#4a4a4a')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style PCR row
        pcr_row = len(table_data) - 1
        for i in range(7):
            table[(pcr_row, i)].set_facecolor('#ffd54f')
            table[(pcr_row, i)].set_text_props(weight='bold')
        
        # Highlight ATM strike
        atm_strike = round(current_price / 50) * 50  # Adjust based on symbol
        for i, strike in enumerate(analysis["strikes"], 1):
            if strike == atm_strike:
                for j in range(7):
                    table[(i, j)].set_facecolor('#e3f2fd')
        
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor='white', bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

# ======================== TELEGRAM ALERTER ========================
class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
    
    async def send_chart(self, chart_buffer: BytesIO, symbol: str):
        """Send chart image to Telegram"""
        try:
            caption = f"📊 {symbol} | {datetime.now(IST).strftime('%d-%b %H:%M')}"
            
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
        """Check if market is open"""
        now = datetime.now(IST).time()
        return MARKET_START <= now <= MARKET_END
    
    async def process_symbols(self):
        """Process all symbols"""
        now_time = datetime.now(IST)
        
        if not self.is_market_open():
            logger.info(f"⏸️ Market closed | Current time: {now_time.strftime('%H:%M:%S IST')}")
            logger.info(f"   Market hours: {MARKET_START} - {MARKET_END}")
            logger.info(f"   Next market open: Monday 09:15 AM\n")
            return
        
        logger.info("\n" + "="*60)
        logger.info(f"🔍 ANALYSIS CYCLE - {datetime.now(IST).strftime('%H:%M:%S')}")
        logger.info("="*60)
        
        # Priority: Indices first, then stocks
        all_symbols = INDICES + NIFTY50_STOCKS
        
        for symbol in all_symbols:
            try:
                analysis = await self.analyzer.analyze_symbol(symbol)
                
                if analysis:
                    # Generate chart
                    chart = self.chart_gen.create_combined_chart(analysis)
                    
                    # Send to Telegram
                    await self.alerter.send_chart(chart, symbol)
                    
                    logger.info(f"✅ {symbol} complete\n")
                else:
                    logger.warning(f"⚠️ {symbol} failed\n")
                
                # Delay between symbols
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}\n")
        
        logger.info("="*60)
        logger.info("✅ CYCLE COMPLETE")
        logger.info("="*60 + "\n")
    
    async def run(self):
        """Main bot loop"""
        # Check current market status
        current_time = datetime.now(IST)
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        
        # Flush startup message immediately
        print("\n" + "="*60, flush=True)
        print("🚀 UPSTOX OPTIONS BOT STARTED!", flush=True)
        print("="*60, flush=True)
        print(f"📅 Date: {current_time.strftime('%d-%b-%Y %A')}", flush=True)
        print(f"🕐 Time: {current_time.strftime('%H:%M:%S IST')}", flush=True)
        print(f"📊 Market Status: {market_status}", flush=True)
        print(f"⏱️  Analysis Interval: {ANALYSIS_INTERVAL // 60} minutes", flush=True)
        print(f"📊 Symbols: {len(INDICES)} indices + {len(NIFTY50_STOCKS)} stocks", flush=True)
        print(f"🕐 Market Hours: {MARKET_START} - {MARKET_END}", flush=True)
        print(f"🎯 ATM Range: ±{ATM_RANGE} strikes", flush=True)
        print("="*60 + "\n", flush=True)
        
        await self.client.create_session()
        
        logger.info("\n" + "="*60)
        logger.info("🚀 UPSTOX OPTIONS BOT STARTED!")
        logger.info("="*60)
        logger.info(f"📅 Date: {current_time.strftime('%d-%b-%Y %A')}")
        logger.info(f"🕐 Time: {current_time.strftime('%H:%M:%S IST')}")
        logger.info(f"📊 Market Status: {market_status}")
        logger.info(f"⏱️  Interval: {ANALYSIS_INTERVAL // 60} minutes")
        logger.info(f"📊 Symbols: {len(INDICES)} indices + {len(NIFTY50_STOCKS)} stocks")
        logger.info(f"🕐 Market Hours: {MARKET_START} - {MARKET_END}")
        logger.info(f"🎯 ATM Range: ±{ATM_RANGE} strikes")
        logger.info("="*60 + "\n")
        
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
