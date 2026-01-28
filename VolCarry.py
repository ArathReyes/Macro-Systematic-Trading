import pandas as pd
import numpy as np
import scipy.stats as si



def garman_kohlhagen(S, K, T, r_d, r_f, sigma, option_type='call'):
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S), 0
    
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * np.exp(-r_f * T) * si.norm.cdf(d1) - K * np.exp(-r_d * T) * si.norm.cdf(d2)
        delta = np.exp(-r_f * T) * si.norm.cdf(d1)
    elif option_type == 'put':
        price = K * np.exp(-r_d * T) * si.norm.cdf(-d2) - S * np.exp(-r_f * T) * si.norm.cdf(-d1)
        delta = -np.exp(-r_f * T) * si.norm.cdf(-d1)
        
    return price, delta

class VolCarryBacktester:
    def __init__(self, data, tenor='1M', notional=100_000, delta_hedge=True, 
                 option_spread_pct=0.02, spot_spread_pips=0.0010):
        """
        option_spread_pct: Bid-Ask spread as % of premium. 
                           Example: 0.02 means 2% width (Sell at 99%, Buy at 101%).
        spot_spread_pips:  Cost to trade spot for hedging (in price terms).
                           Example: 0.0010 is 10 pips (common for USDMXN institutional size).
        """
        self.df = data.copy()
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        self.notional = notional
        self.delta_hedge = delta_hedge
        
        # Cost Parameters
        self.opt_spread = option_spread_pct
        self.spot_spread = spot_spread_pips
        
        tenor_map = {'1M': 30, '3M': 90, '6M': 180, '1Y': 360}
        self.dte = tenor_map.get(tenor, 30)
        
    def run(self):
        results = []
        current_hedge_shares = 0 # Track how much spot we hold
        
        for i in range(len(self.df) - 1):
            # --- 1. SETUP (Today t0) ---
            date = self.df.index[i]
            next_date = self.df.index[i+1]
            
            S0 = self.df.iloc[i]['Spot']
            sigma0 = self.df.iloc[i]['IV']
            rd = self.df.iloc[i]['Local_Rate']
            rf = self.df.iloc[i]['USD_Rate']
            
            # Calculate Forward & Strike (ATM Forward)
            T_year = self.dte / 365.0
            F0 = S0 * np.exp((rd - rf) * T_year)
            K = F0 
            
            # --- 2. EXECUTION (Sell Straddle) ---
            c0, c_delta = garman_kohlhagen(S0, K, T_year, rd, rf, sigma0, 'call')
            p0, p_delta = garman_kohlhagen(S0, K, T_year, rd, rf, sigma0, 'put')
            
            mid_premium = c0 + p0
            
            # Apply Transaction Cost (Selling -> Hit Bid -> Receive Less)
            premium_collected = mid_premium * (1 - self.opt_spread)
            
            # --- 3. HEDGE REBALANCING ---
            hedge_cost = 0
            if self.delta_hedge:
                # Straddle Net Delta (should be near 0 if K=Fwd, but varies)
                option_net_delta = c_delta + p_delta 
                
                # To be delta neutral, we need: Spot_Pos + Option_Delta = 0
                # Target Spot Shares = -Option_Delta * Notional
                target_shares = -option_net_delta * self.notional
                
                # Calculate change required
                shares_to_trade = target_shares - current_hedge_shares
                
                # Cost: We pay half the spread on the volume traded
                hedge_cost = abs(shares_to_trade) * (self.spot_spread / 2)
                
                # Update position tracker
                current_hedge_shares = target_shares
            
            # --- 4. MARK TO MARKET (Tomorrow t1) ---
            S1 = self.df.iloc[i+1]['Spot']
            sigma1 = self.df.iloc[i+1]['IV']
            dt = (next_date - date).days / 365.0
            T_new = T_year - dt
            
            # Buy Back Cost (Buying -> Lift Offer -> Pay More)
            c1, _ = garman_kohlhagen(S1, K, T_new, rd, rf, sigma1, 'call')
            p1, _ = garman_kohlhagen(S1, K, T_new, rd, rf, sigma1, 'put')
            
            mid_value_t1 = c1 + p1
            buy_back_cost = mid_value_t1 * (1 + self.opt_spread)
            
            # --- 5. PnL CALCULATION ---
            # Option PnL
            option_pnl = (premium_collected - buy_back_cost) * self.notional
            
            # Hedge PnL (Directional movement of the hedge shares held overnight)
            hedge_pnl = 0
            if self.delta_hedge:
                hedge_pnl = current_hedge_shares * (S1 - S0)
            
            # Total Daily PnL
            total_daily_pnl = option_pnl + hedge_pnl - hedge_cost
            
            results.append({
                'Date': next_date,
                'Spot': S1,
                'Option_PnL': option_pnl,
                'Hedge_PnL': hedge_pnl,
                'Hedge_Cost': hedge_cost,
                'Total_PnL': total_daily_pnl
            })
            
        self.results_df = pd.DataFrame(results)
        self.results_df.set_index('Date', inplace=True)
        self.results_df['Cumulative_PnL'] = self.results_df['Total_PnL'].cumsum()
        return self.results_df
