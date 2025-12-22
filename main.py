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
UPSTOX_API_V3_URL = "https://api.upstox.com/v3"
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
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
    async def create_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(headers=self.headers)
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    def get_instrument_key(self, symbol: str) -> str:
        """Get Upstox instrument key for underlying"""
        mapping = {
            "NIFTY": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank",
            "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
            "FINNIFTY": "NSE_INDEX|Nifty Fin Service"
        }
        return mapping.get(symbol, f"NSE_EQ|{symbol}")
    
    async def get_ltp(self, instrument_key: str) -> float:
        """Get Last Traded Price"""
        try:
            url = f"{UPSTOX_API_URL}/market-quote/ltp"
            params = {"instrument_key": instrument_key}
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("status") == "success" and data.get("data"):
                    for key, value in data["data"].items():
                        return value.get("last_price", 0.0)
                
                return 0.0
        except Exception as e:
            logger.error(f"❌ Error fetching LTP: {e}")
            return 0.0
    
    async def get_historical_candles(self, instrument_key: str, count: int = 50) -> pd.DataFrame:
        """Get historical intraday candles (5 min) using V3 API"""
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
        """Get all option contracts for symbol and expiry using Upstox API"""
        try:
            instrument_key = self.get_instrument_key(symbol)
            url = f"{UPSTOX_API_URL}/option/contract"
            
            params = {
                "instrument_key": instrument_key,
                "expiry_date": expiry
            }
            
            logger.debug(f"   📡 Fetching option contracts")
            logger.debug(f"      Underlying: {instrument_key}")
            logger.debug(f"      Expiry: {expiry}")
            
            async with self.session.get(url, params=params) as response:
                response_text = await response.text()
                data = json.loads(response_text)
                
                if data.get("status") == "success":
                    contracts = data.get("data", [])
                    
                    if contracts:
                        logger.info(f"   ✅ Fetched {len(contracts)} option contracts")
                        return contracts
                    else:
                        # Try with next expiry if no contracts found
                        logger.warning(f"   ⚠️ No contracts for {expiry}, this might not be a valid expiry")
                        return []
                else:
                    logger.error(f"   ❌ API Error: {data}")
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error fetching option contracts: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def get_expiries(self, symbol: str) -> List[str]:
        """Get available expiries for symbol - returns next 2 Thursdays"""
        try:
            today = datetime.now(IST).date()
            expiries = []
            
            # Find next Thursday
            days_ahead = (3 - today.weekday()) % 7  # Thursday = 3
            if days_ahead == 0:
                days_ahead = 7
            
            # Get next 2 Thursdays (weekly expiries)
            for i in range(2):
                next_expiry = today + timedelta(days=days_ahead + (i * 7))
                expiries.append(next_expiry.strftime("%Y-%m-%d"))
            
            logger.debug(f"   📅 Available expiries: {expiries}")
            return expiries
            
        except Exception as e:
            logger.error(f"❌ Error calculating expiry: {e}")
            return []

# ======================== OPTION CHAIN ANALYZER ========================
class OptionAnalyzer:
    def __init__(self, client: UpstoxClient):
        self.client = client
    
    def get_strike_interval(self, symbol: str) -> int:
        """Get strike interval for symbol"""
        intervals = {
            "NIFTY": 50,
            "BANKNIFTY": 100,
            "MIDCPNIFTY": 25,
            "FINNIFTY": 50
        }
        return intervals.get(symbol, 100)
    
    async def filter_atm_strikes(self, contracts: List[Dict], current_price: float, symbol: str) -> Dict:
        """Filter contracts to get ATM ±5 strikes with their data"""
        interval = self.get_strike_interval(symbol)
        
        # Calculate ATM strike
        atm = round(current_price / interval) * interval
        
        # Calculate strike range
        min_strike = atm - (ATM_RANGE * interval)
        max_strike = atm + (ATM_RANGE * interval)
        
        logger.info(f"   🎯 ATM: {atm}, Range: {min_strike} to {max_strike}")
        
        # Separate CE and PE contracts
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
        """Fetch LTP for all option contracts"""
        # Fetch CE prices
        for strike, contract in contracts_data["ce"].items():
            ltp = await self.client.get_ltp(contract["instrument_key"])
            contract["ltp"] = ltp
            await asyncio.sleep(0.05)  # Rate limiting
        
        # Fetch PE prices
        for strike, contract in contracts_data["pe"].items():
            ltp = await self.client.get_ltp(contract["instrument_key"])
            contract["ltp"] = ltp
            await asyncio.sleep(0.05)  # Rate limiting
    
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
            
            # Try each expiry until we find contracts
            contracts = []
            expiry = None
            
            for exp_date in expiries:
                logger.info(f"   📅 Trying expiry: {exp_date}")
                contracts = await self.client.get_option_contracts(symbol, exp_date)
                
                if contracts:
                    expiry = exp_date
                    logger.info(f"   ✅ Using expiry: {expiry}")
                    break
                else:
                    logger.warning(f"   ⚠️ No contracts for {exp_date}, trying next...")
            
            if not contracts or not expiry:
                logger.warning(f"⚠️ No valid option contracts found for {symbol}")
                return None
            
            # Filter ATM ±5 strikes
            contracts_data = await self.filter_atm_strikes(contracts, current_price, symbol)
            
            if not contracts_data["strikes"]:
                logger.warning(f"⚠️ No strikes found in ATM range")
                return None
            
            logger.info(f"   🎯 Strikes: {contracts_data['strikes'][0]} to {contracts_data['strikes'][-1]}")
            
            # Fetch prices for all strikes
            await self.fetch_option_prices(contracts_data)
            
            # Calculate PCR (OI data not available in LTP endpoint, will be 0)
            ce_data = [contracts_data["ce"].get(s, {"strike": s, "ltp": 0, "oi": 0, "volume": 0}) for s in contracts_data["strikes"]]
            pe_data = [contracts_data["pe"].get(s, {"strike": s, "ltp": 0, "oi": 0, "volume": 0}) for s in contracts_data["strikes"]]
            
            total_ce_oi = sum(d["oi"] for d in ce_data)
            total_pe_oi = sum(d["oi"] for d in pe_data)
            
            pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
            pcr_vol = 0  # Volume data not available
            
            logger.info(f"   📊 CE count: {len(ce_data)}, PE count: {len(pe_data)}")
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "expiry": expiry,
                "candles": candles,
                "strikes": contracts_data["strikes"],
                "ce_data": ce_data,
                "pe_data": pe_data,
                "pcr_oi": pcr_oi,
                "pcr_vol": pcr_vol
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        
        # Add PCR row (OI will be N/A)
        table_data.append([
            "PCR",
            "",
            "",
            f"OI: N/A",
            f"Vol: N/A",
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
        interval = 50 if symbol == "NIFTY" else 100
        atm_strike = round(current_price / interval) * interval
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
        today = datetime.now(IST).date()
        
        # Check if weekend
        if today.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
            
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
        
        test_symbols = INDICES
        logger.info(f"📊 Analyzing {len(test_symbols)} indices\n")
        
        for symbol in test_symbols:
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
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("="*60)
        logger.info("✅ CYCLE COMPLETE")
        logger.info("="*60 + "\n")
    
    async def run(self):
        """Main bot loop"""
        current_time = datetime.now(IST)
        market_status = "🟢 OPEN" if self.is_market_open() else "🔴 CLOSED"
        
        print("\n" + "="*60, flush=True)
        print("🚀 UPSTOX OPTIONS BOT STARTED!", flush=True)
        print("="*60, flush=True)
        print(f"📅 Date: {current_time.strftime('%d-%b-%Y %A')}", flush=True)
        print(f"🕐 Time: {current_time.strftime('%H:%M:%S IST')}", flush=True)
        print(f"📊 Market Status: {market_status}", flush=True)
        print(f"⏱️  Analysis Interval: {ANALYSIS_INTERVAL // 60} minutes", flush=True)
        print(f"📊 Symbols: {len(INDICES)} indices", flush=True)
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
        logger.info(f"📊 Symbols: {len(INDICES)} indices")
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
                    import traceback
                    logger.error(traceback.format_exc())
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
