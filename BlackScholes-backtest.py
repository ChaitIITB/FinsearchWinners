import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, q, sigma, option_type='call'):
    """
    Black-Scholes price for European options with continuous dividend yield.

    Parameters:
    S : float or np.array  -> Spot price
    K : float or np.array  -> Strike price
    T : float or np.array  -> Time to maturity (in years)
    r : float              -> Risk-free rate (annualized, decimal)
    q : float              -> Continuous dividend yield (annualized, decimal)
    sigma : float or np.array -> Volatility (annualized, decimal)
    option_type : 'call' or 'put'

    Returns:
    price : float or np.array
    """

    S, K, T, sigma = map(np.array, (S, K, T, sigma))

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def black_scholes_greeks(S, K, T, r, q, sigma, option_type='call'):
    """
    Compute Greeks for European options (Black-Scholes with dividend yield).
    Returns dict with delta, gamma, vega, theta, rho.
    """

    S, K, T, sigma = map(np.array, (S, K, T, sigma))

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = None
    if option_type == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
    elif option_type == 'put':
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    if option_type == 'call':
        theta = (- (S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        theta = (- (S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


# Example usage:
if __name__ == "__main__":
    S = 100      # Spot price
    K = 105      # Strike price
    T = 0.5      # 6 months to expiry
    r = 0.05     # 5% risk-free rate
    q = 0.02     # 2% dividend yield
    sigma = 0.25 # 25% volatility

    call_price = black_scholes(S, K, T, r, q, sigma, 'call')
    put_price = black_scholes(S, K, T, r, q, sigma, 'put')
    call_greeks = black_scholes_greeks(S, K, T, r, q, sigma, 'call')

    print(f"Call Price: {call_price:.4f}")
    print(f"Put Price: {put_price:.4f}")
    print("Call Greeks:", call_greeks)























import pandas as pd
import numpy as np

# Technical analysis
import talib as ta

# Datetime manipulation
from datetime import timedelta

# Ignore warnings
import warnings
warnings.simplefilter('ignore')

# Helper functions
import sys
sys.path.append('..')
from data_modules.options_util_quantra import get_IV_percentile, get_premium, get_expected_profit_empirical, setup_butterfly


# Read data
options_data = pd.read_pickle('../data_modules/nifty_options_data_2019_2022.bz2')
data = pd.read_pickle('../data_modules/nifty_data_2019_2022.bz2')
data.head()

# Calculate ADX
data['ADX'] = ta.ADX(data.spot_high, data.spot_low, data.spot_close, timeperiod=14)

# Calculate IVP
data['IVP'] = get_IV_percentile(data, options_data, window = 60)

# Calculate days to expiry
data['days_to_expiry'] = (data['Expiry'] - data.index).dt.days

# IVP entry condition
condition_1 = (data['IVP'] >= 50) & (data['IVP'] <= 95)

# ADX entry condition
condition_2 = (data['ADX'] <= 30)

# Generate signal as 1 when both conditions are true
data['signal_adx_ivp'] = np.where(condition_1 & condition_2, 1, np.nan)

# Generate signal as 0 on expiry dates
data['signal_adx_ivp'] = np.where(data.index == data.Expiry, 0, data['signal_adx_ivp'])

# Display bottom 5 rows
data.tail()


# Backtesting
# Create dataframes for round trips, storing trades, and mtm
round_trips_details = pd.DataFrame()
trades = pd.DataFrame()
mark_to_market = pd.DataFrame()

# Function for calculating mtm
def add_to_mtm(mark_to_market, option_strategy, trading_date):
    option_strategy['Date'] = trading_date
    mark_to_market = pd.concat([mark_to_market, option_strategy])
    return mark_to_market

# Initialise current position, number of trades and cumulative pnl to 0
current_position = 0
trade_num = 0
cum_pnl = 0

# Set exit flag to False
exit_flag = False

# Set start date for backtesting
start_date = data.index[0] + timedelta(days=90) 


for i in data.loc[start_date:].index:

    if (current_position == 0) & (data.loc[i, 'signal_adx_ivp'] == 1):
        
        # Setup butterfly
        options_data_daily = options_data.loc[i]
        butterfly = setup_butterfly(data.loc[i,'futures_close'], options_data_daily, direction = "short") 

        # List of all strike prices        
        price_range = list(options_data_daily['Strike Price'].unique())        
  
        # start_date for fetching historical data
        start_date = i - timedelta(days=90)        
        
        # Calculate Expected profit        
        data.loc[i,'exp_profit'] = get_expected_profit_empirical(data.loc[start_date:i], 
                                    butterfly.copy(), data.loc[i, 'days_to_expiry'], price_range)
        
        # We are going against the historical data 
        if data.loc[i,'exp_profit'] < 0:
            
            # Check that the last price of all the legs of the butterfly is greater than 0
            if (butterfly.premium.isna().sum() > 0) or ((butterfly.premium == 0).sum() > 0):
                print(f"\x1b[31mStrike price is not liquid so we will ignore this trading opportunity {i}\x1b[0m")
                continue
            
            # Populate the trades dataframe
            trades = butterfly.copy()
            trades['entry_date'] = i
            trades.rename(columns={'premium':'entry_price'}, inplace=True)            
            
            # Calculate net premium 
            net_premium = round((butterfly.position * butterfly.premium).sum(),1)
            
            # Update current position to 1
            current_position = 1
            
            # Update mark_to_market dataframe
            mark_to_market = add_to_mtm(mark_to_market, butterfly, i)
            
            # Increase number of trades by 1
            trade_num += 1   
            print("-"*30)
            
            # Print trade details
            print(f"Trade No: {trade_num} | Entry | Date: {i} | Premium: {net_premium}")            
            
    elif current_position == 1:
        
        # Update net premium
        options_data_daily = options_data.loc[i]
        butterfly['premium'] = butterfly.apply(lambda r: get_premium(r, options_data_daily), axis=1)        
        net_premium = (butterfly.position * butterfly.premium).sum()
        
        # Update mark_to_market dataframe
        mark_to_market = add_to_mtm(mark_to_market, butterfly, i)
      
        # Exit at expiry
        if data.loc[i, 'signal_adx_ivp'] == 0:
            exit_type = 'Expiry'
            exit_flag = True            
        
            
        if exit_flag:
            
            # Check that the data is present for all strike prices on the exit date
            if butterfly.premium.isna().sum() > 0:
                print(f"Data missing for the required strike prices on {i}, Not adding to trade logs.")
                current_position = 0
                continue
            
            # Update the trades dataframe
            trades['exit_date'] = i
            trades['exit_type'] = exit_type
            trades['exit_price'] = butterfly.premium
            
            # Add the trade logs to round trip details
            round_trips_details = pd.concat([round_trips_details,trades])
            
            # Calculate net premium at exit
            net_premium = round((butterfly.position * butterfly.premium).sum(),1)   
            
            # Calculate net premium on entry
            entry_net_premium = (trades.position * trades.entry_price).sum()       
            
            # Calculate pnl for the trade
            trade_pnl = round(net_premium - entry_net_premium,1)
            
            # Calculate cumulative pnl
            cum_pnl += trade_pnl
            cum_pnl = round(cum_pnl,2)
            
            # Print trade details
            print(f"Trade No: {trade_num} | Exit Type: {exit_type} | Date: {i} | Premium: {net_premium} | PnL: {trade_pnl} | Cum PnL: {cum_pnl}")                              

            # Update current position to 0
            current_position = 0    
            
            # Set exit flag to false
            exit_flag = False          


# Round trip details
round_trips_details.head()


mark_to_market.head(7)


