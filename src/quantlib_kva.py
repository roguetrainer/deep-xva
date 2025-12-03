import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt

def calculate_kva():
    print("--- Starting QuantLib KVA Simulation ---")
    
    # 1. SETUP THE MARKET
    today = ql.Date(1, 12, 2025)
    ql.Settings.instance().evaluationDate = today

    # Flat yield curve (5% risk-free rate)
    rate = ql.SimpleQuote(0.05)
    rate_handle = ql.QuoteHandle(rate)
    yield_curve = ql.FlatForward(today, rate_handle, ql.Actual360())
    yield_handle = ql.YieldTermStructureHandle(yield_curve)

    # 2. DEFINE THE INSTRUMENT (SWAP)
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    start_date = calendar.advance(today, 2, ql.Days)
    
    nominal = 1_000_000
    fixed_rate = 0.05 # At-the-money
    index = ql.USDLibor(ql.Period("3M"), yield_handle)

    swap = ql.MakeVanillaSwap(ql.Period("5Y"), index, fixed_rate, ql.Period("3M"))
    swap.setPricingEngine(ql.DiscountingSwapEngine(yield_handle))

    print(f"Initial Swap NPV: {swap.NPV():.2f}")

    # 3. SIMULATION (Hull-White Model)
    sigma = 0.01  # Volatility
    a = 0.03      # Mean reversion
    hw_process = ql.HullWhiteProcess(yield_handle, a, sigma)

    # Simulation parameters
    num_paths = 500        # Lower for demo speed
    time_steps = 60        # Monthly steps for 5 years
    sequence_generator = ql.UniformRandomSequenceGenerator(
        time_steps, ql.UniformRandomGenerator())
    gaussian_sequence_generator = ql.GaussianRandomSequenceGenerator(
        sequence_generator)
    path_generator = ql.GaussianPathGenerator(
        hw_process, 5.0, time_steps, gaussian_sequence_generator, False)

    future_values = np.zeros((time_steps, num_paths))
    times = np.linspace(0, 5.0, time_steps)

    # 4. RUN MONTE CARLO
    print(f"Simulating {num_paths} paths over {time_steps} steps...")
    
    for i in range(num_paths):
        sample_path = path_generator.next()
        path = sample_path.value()
        
        for t_idx in range(time_steps):
            time_val = times[t_idx]
            rate_at_t = path[t_idx]
            
            # PROXY PRICING (Fast approx for demo)
            # In real KVA, this line requires a nested simulation
            remaining_time = 5.0 - time_val
            if remaining_time < 0: remaining_time = 0
            
            df = np.exp(-0.05 * remaining_time)
            duration = (1 - np.exp(-0.05 * remaining_time)) / 0.05
            
            # Value ~ (Fixed - Rate_at_t) * Duration * Notional
            val = (fixed_rate - rate_at_t) * duration * nominal * df
            future_values[t_idx, i] = val

    # 5. CALCULATE KVA
    # EPE (Expected Positive Exposure)
    positive_exposure = np.maximum(future_values, 0)
    expected_exposure = np.mean(positive_exposure, axis=1)

    # Proxy Capital = 8% of Risk Weighted Assets (assumed equal to Exposure)
    capital_requirement = expected_exposure * 0.08 

    # KVA Cost = Integral of (Capital * Cost of Capital * DF)
    cost_of_capital = 0.10  # 10% ROE
    dt = 5.0 / time_steps
    discount_factors = np.exp(-0.05 * times)

    kva_time_buckets = capital_requirement * cost_of_capital * discount_factors * dt
    total_kva = np.sum(kva_time_buckets)

    print(f"Total KVA Charge: ${total_kva:.2f}")

    # 6. VISUALIZATION
    plt.figure(figsize=(10, 6))
    plt.plot(times, expected_exposure, label="Expected Exposure (EPE)")
    plt.plot(times, capital_requirement, label="Regulatory Capital Profile")
    plt.fill_between(times, capital_requirement, alpha=0.3, color='orange', label="Capital Held")
    plt.title(f"KVA Profile (Total KVA = ${total_kva:,.2f})")
    plt.xlabel("Years")
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    calculate_kva()
