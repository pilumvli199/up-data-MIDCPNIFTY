"""
Upstox Option Chain + Chart Telegram Bot
5-minute TF candlestick + ±5 ATM strikes
Single PNG with chart + option data
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import schedule
from dotenv import load_dotenv

load_dotenv()

# ===================== CONFIG =====================
UPSTOX_API_KEY = os.getenv('UPSTOX_API_KEY')
UPSTOX_ACCESS_TOKEN = os.getenv('UPSTOX_ACCESS_TOKEN')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

BASE_URL = "https://api.upstox.com/v2"
IST = pytz.timezone('Asia/Kolkata')

# Symbols to track
INDICES = ['NIFTY', 'BANKNIFTY', 'MIDCPNIFTY', 'FINNIFTY']

NIFTY50_STOCKS = [
    'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
    'BAJAJ_AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BPCL', 'BHARTIARTL',
    'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY',
    'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
    'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC',
    'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LT',
    'M&M', 'MARUTI', 'NTPC', 'NESTLEIND', 'ONGC',
    'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SUNPHARMA',
    'TCS', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TECHM',
    'TITAN', 'ULTRACEMCO', 'UPL', 'WIPRO', 'LTIM'
]

ALL_SYMBOLS = INDICES + NIFTY50_STOCKS


# ===================== UPSTOX API FUNCTIONS =====================

def get_instrument_key(symbol):
    """Convert symbol to Upstox instrument key"""
    if symbol in INDICES:
        return f"NSE_INDEX|{symbol}"
    else:
        return f"NSE_EQ|{symbol}"


def get_option_chain(symbol, expiry_date):
    """Fetch option chain from Upstox"""
    url = f"{BASE_URL}/option/chain"
    
    headers = {
        'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}',
        'Accept': 'application/json'
    }
    
    params = {
        'instrument_key': get_instrument_key(symbol),
        'expiry_date': expiry_date
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Option Chain Error ({symbol}): {e}")
        return None


def get_historical_candles(symbol, interval='5minute', days_back=1):
    """Fetch historical candlestick data (last 50 candles)"""
    
    # Get current date in IST
    to_date = datetime.now(IST).strftime('%Y-%m-%d')
    
    url = f"{BASE_URL}/historical-candle/{get_instrument_key(symbol)}/{interval}/{to_date}"
    
    headers = {
        'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'success' and 'data' in data and 'candles' in data['data']:
            candles = data['data']['candles'][:50]  # Last 50 candles
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
        else:
            print(f"❌ No candle data for {symbol}")
            return None
            
    except Exception as e:
        print(f"❌ Historical Data Error ({symbol}): {e}")
        return None


def get_ltp(symbol):
    """Get Last Traded Price"""
    url = f"{BASE_URL}/market-quote/ltp"
    
    headers = {
        'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}',
        'Accept': 'application/json'
    }
    
    params = {
        'instrument_key': get_instrument_key(symbol)
    }
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'success':
            return data['data'][get_instrument_key(symbol)]['last_price']
        return None
    except Exception as e:
        print(f"❌ LTP Error ({symbol}): {e}")
        return None


def get_nearest_expiry():
    """Get nearest weekly expiry (Thursday for NIFTY/BANKNIFTY)"""
    today = datetime.now(IST).date()
    days_ahead = (3 - today.weekday()) % 7  # Thursday = 3
    
    if days_ahead == 0 and datetime.now(IST).hour >= 15:  # After 3:30 PM
        days_ahead = 7
    
    expiry = today + timedelta(days=days_ahead)
    return expiry.strftime('%Y-%m-%d')


def process_option_chain_data(symbol):
    """Process option chain and extract ±5 ATM strikes"""
    
    # Get LTP for ATM calculation
    ltp = get_ltp(symbol)
    if not ltp:
        return None
    
    # Get option chain
    expiry = get_nearest_expiry()
    chain_data = get_option_chain(symbol, expiry)
    
    if not chain_data or 'data' not in chain_data:
        return None
    
    # Find ATM strike
    strike_interval = 50 if symbol in ['NIFTY', 'FINNIFTY'] else 100
    atm_strike = round(ltp / strike_interval) * strike_interval
    
    # Extract ±5 strikes
    strikes_to_fetch = [atm_strike + (i * strike_interval) for i in range(-5, 6)]
    
    option_data = []
    
    for strike in strikes_to_fetch:
        ce_data = pe_data = None
        
        for item in chain_data['data']:
            if item['strike_price'] == strike:
                if item['option_type'] == 'CE':
                    ce_data = item
                elif item['option_type'] == 'PE':
                    pe_data = item
        
        if ce_data and pe_data:
            option_data.append({
                'Strike': strike,
                'CE_OI': ce_data.get('open_interest', 0),
                'CE_Volume': ce_data.get('volume', 0),
                'CE_LTP': ce_data.get('last_price', 0),
                'PE_LTP': pe_data.get('last_price', 0),
                'PE_Volume': pe_data.get('volume', 0),
                'PE_OI': pe_data.get('open_interest', 0)
            })
    
    df = pd.DataFrame(option_data)
    
    if not df.empty:
        # Calculate PCR
        total_ce_oi = df['CE_OI'].sum()
        total_pe_oi = df['PE_OI'].sum()
        pcr_oi = round(total_pe_oi / total_ce_oi, 2) if total_ce_oi > 0 else 0
        
        total_ce_vol = df['CE_Volume'].sum()
        total_pe_vol = df['PE_Volume'].sum()
        pcr_vol = round(total_pe_vol / total_ce_vol, 2) if total_ce_vol > 0 else 0
        
        return {
            'symbol': symbol,
            'ltp': ltp,
            'atm_strike': atm_strike,
            'data': df,
            'pcr_oi': pcr_oi,
            'pcr_volume': pcr_vol
        }
    
    return None


# ===================== CHART GENERATION =====================

def create_candlestick_only_chart(symbol, candle_df):
    """Create candlestick chart only (for stocks without options)"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    
    # Draw candlesticks
    for idx, row in candle_df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # Candle body
        height = abs(row['close'] - row['open'])
        bottom = min(row['open'], row['close'])
        ax.bar(idx, height, bottom=bottom, width=0.6, color=color, alpha=0.8)
        
        # Wicks
        ax.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=0.8)
    
    # Styling
    ax.set_title(f"{symbol} - 5 Min Candlestick Chart", fontsize=16, fontweight='bold', color='black')
    ax.set_xlabel('Time', fontsize=12, color='black')
    ax.set_ylabel('Price', fontsize=12, color='black')
    ax.grid(True, alpha=0.3, color='gray')
    ax.tick_params(colors='black')
    
    # Add current price
    current_price = candle_df.iloc[-1]['close']
    ax.text(0.98, 0.98, f"LTP: ₹{current_price:.2f}", 
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    # Save to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


def create_combined_chart(symbol, candle_df, option_data):
    """Create single PNG with candlestick + option chain"""
    
    # Set white background style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = GridSpec(3, 1, height_ratios=[2.5, 1, 0.3], hspace=0.3)
    
    # ========== CANDLESTICK CHART ==========
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('white')
    
    for idx, row in candle_df.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # Candle body
        height = abs(row['close'] - row['open'])
        bottom = min(row['open'], row['close'])
        ax1.bar(idx, height, bottom=bottom, width=0.6, color=color, alpha=0.8)
        
        # Wicks
        ax1.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=0.8)
    
    ax1.set_title(f"{symbol} - 5 Min Candlestick Chart", fontsize=16, fontweight='bold', color='black')
    ax1.set_ylabel('Price', fontsize=12, color='black')
    ax1.grid(True, alpha=0.3, color='gray')
    ax1.tick_params(colors='black')
    
    # ========== OPTION CHAIN TABLE ==========
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('tight')
    ax2.axis('off')
    
    # Prepare table data
    table_data = []
    headers = ['Strike', 'CE OI', 'CE Vol', 'CE LTP', 'PE LTP', 'PE Vol', 'PE OI']
    
    for _, row in option_data['data'].iterrows():
        strike_color = 'lightgreen' if row['Strike'] == option_data['atm_strike'] else 'white'
        table_data.append([
            f"{int(row['Strike'])}",
            f"{int(row['CE_OI']):,}",
            f"{int(row['CE_Volume']):,}",
            f"{row['CE_LTP']:.2f}",
            f"{row['PE_LTP']:.2f}",
            f"{int(row['PE_Volume']):,}",
            f"{int(row['PE_OI']):,}"
        ])
    
    table = ax2.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.12, 0.13, 0.13, 0.12, 0.12, 0.13, 0.13])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_facecolor('#4472C4')
            cell.set_text_props(weight='bold', color='white')
        else:
            if option_data['data'].iloc[i-1]['Strike'] == option_data['atm_strike']:
                cell.set_facecolor('lightgreen')
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor('gray')
    
    # ========== PCR INFO BOX ==========
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    info_text = f"""LTP: ₹{option_data['ltp']:.2f}  |  ATM: {option_data['atm_strike']}  |  PCR (OI): {option_data['pcr_oi']}  |  PCR (Volume): {option_data['pcr_volume']}"""
    
    ax3.text(0.5, 0.5, info_text, ha='center', va='center', 
            fontsize=12, fontweight='bold', color='black',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
    
    # Save to BytesIO
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf


# ===================== TELEGRAM FUNCTIONS =====================

def send_telegram_photo(photo_buffer, symbol):
    """Send PNG to Telegram"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    timestamp = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')
    
    files = {
        'photo': (f'{symbol}_{timestamp}.png', photo_buffer, 'image/png')
    }
    
    data = {
        'chat_id': TELEGRAM_CHAT_ID
    }
    
    try:
        response = requests.post(url, files=files, data=data, timeout=30)
        response.raise_for_status()
        print(f"✅ Sent {symbol} chart to Telegram")
        return True
    except Exception as e:
        print(f"❌ Telegram Error ({symbol}): {e}")
        return False


# ===================== MAIN PROCESSING =====================

def process_symbol(symbol):
    """Process single symbol: fetch data + generate chart + send to Telegram"""
    
    print(f"\n{'='*50}")
    print(f"🔄 Processing: {symbol}")
    print(f"{'='*50}")
    
    try:
        # Fetch candlestick data
        candle_df = get_historical_candles(symbol)
        if candle_df is None or candle_df.empty:
            print(f"❌ No candle data for {symbol}")
            return False
        
        print(f"✅ Fetched {len(candle_df)} candles")
        
        # Check if symbol has options (only indices)
        if symbol in INDICES:
            # Fetch option chain for indices
            option_data = process_option_chain_data(symbol)
            if option_data is None:
                print(f"❌ No option chain data for {symbol}")
                return False
            
            print(f"✅ Option chain processed - PCR OI: {option_data['pcr_oi']}")
            
            # Generate combined chart (candles + options)
            chart_buffer = create_combined_chart(symbol, candle_df, option_data)
            
            # Send to Telegram
            send_telegram_photo(chart_buffer, symbol)
            
        else:
            # Stocks - only candlestick chart (no options)
            print(f"📊 Stock - Candlestick only (no options)")
            
            # Create simple candlestick chart
            chart_buffer = create_candlestick_only_chart(symbol, candle_df)
            
            # Send to Telegram
            send_telegram_photo(chart_buffer, symbol)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")
        return False


def run_all_symbols():
    """Process all symbols sequentially"""
    
    now = datetime.now(IST)
    
    # Check market hours (9:15 AM - 3:30 PM)
    if not (9 <= now.hour < 15 or (now.hour == 15 and now.minute <= 30)):
        print(f"⏸️ Market closed - {now.strftime('%H:%M:%S')}")
        return
    
    print(f"\n{'#'*60}")
    print(f"🚀 Starting scan at {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")
    
    success_count = 0
    
    for symbol in ALL_SYMBOLS:
        if process_symbol(symbol):
            success_count += 1
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n{'='*60}")
    print(f"✅ Completed: {success_count}/{len(ALL_SYMBOLS)} symbols")
    print(f"{'='*60}\n")


# ===================== SCHEDULER =====================

def start_scheduler():
    """Run every 5 minutes during market hours"""
    
    # Schedule every 5 minutes
    schedule.every(5).minutes.do(run_all_symbols)
    
    print("🤖 Upstox Chart Bot Started!")
    print("📊 Tracking: NIFTY, BANKNIFTY, MIDCPNIFTY, FINNIFTY + 50 stocks")
    print("⏰ Running every 5 minutes (9:15 AM - 3:30 PM)\n")
    
    # Run immediately on start
    run_all_symbols()
    
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    start_scheduler()
