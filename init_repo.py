import os
import json

# --- CONFIGURATION ---
REPO_NAME = "deep-xva"

# --- CONTENT DEFINITIONS ---

README_CONTENT = """# deep-xva

**PyTorch Neural SDE solver for fast XVA/KVA. Solves high-dimensional BSDEs using deep learning to calculate capital profiles without nested simulations.**

## Overview

This repository demonstrates two approaches to calculating Valuation Adjustments (specifically KVA - Capital Valuation Adjustment):

1.  **The "Classic" Approach (src/quantlib_kva.py):** Uses `QuantLib` and a Hull-White model to simulate interest rate paths. It calculates exposure profiles via Monte Carlo simulation. This demonstrates the "nested simulation" bottleneck.
2.  **The "Deep Learning" Approach (src/neural_sde_solver.py):** Uses `PyTorch` to implement a "Deep BSDE" solver. This represents the modern approach where a Neural Network learns the hedging/capital function, bypassing the need for nested simulations.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\\Scripts\\activate on Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run as Scripts
You can run the pure Python scripts directly:

```bash
# Run the Neural SDE Solver (Fast)
python src/neural_sde_solver.py

# Run the QuantLib KVA Calculator (Traditional)
python src/quantlib_kva.py
```

### Run Notebooks
Navigate to the `notebooks/` directory and launch Jupyter:

```bash
jupyter notebook
```
"""

REQUIREMENTS_CONTENT = """torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
QuantLib>=1.30.0
scipy>=1.10.0
jupyter>=1.0.0
"""

THEORY_CONTENT = """# Neural SDEs and the KVA Problem

## The Problem: KVA (Capital Valuation Adjustment)

KVA is the cost of holding regulatory capital against a derivative trade for its entire lifetime.

To calculate it accurately, you need a **Nested Simulation**:
1.  **Outer Loop:** Simulate the market forward (e.g., 1,000 paths for 30 years).
2.  **Inner Loop:** At *every* time step and *every* path of the Outer Loop, you must calculate the "Regulatory Capital."
    * Under Basel III, this often requires calculating "Expected Shortfall," which requires *another* simulation (e.g., 1,000 paths) starting from that future point.

**Total Complexity:** $1,000 \\times 30 \\text{ years} \\times 1,000 = 30,000,000$ valuations. This is computationally prohibitive.

## The Solution: Neural SDEs (Deep BSDE)

A Neural SDE (Stochastic Differential Equation) replaces the "Inner Loop" with a Neural Network.

We treat the **Capital Requirement** (or the Option Price) and the **Hedging Strategy** as unknown functions to be learned.

1.  We define a Neural Network $Z(t, X_t)$ that takes the current time and market state and predicts the Hedge/Gradient.
2.  We run *only* the Outer Loop.
3.  We use the output of the Neural Network to step the portfolio value forward in time.
4.  **Loss Function:** We force the final portfolio value to equal the Regulatory Payoff. Backpropagation updates the Neural Network to correct the hedge.

**Result:** The Neural Network "learns" the entire capital surface. Once trained, getting the capital requirement at any future time takes milliseconds (one inference) rather than hours (nested Monte Carlo).
"""

NEURAL_SDE_SOURCE = r"""import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm

# --- CONFIGURATION ---
BATCH_SIZE = 64      # Number of paths simulated at once
DIMENSION = 1        # Number of assets
TOTAL_TIME = 1.0     # T = 1 year
NUM_STEPS = 50       # N time steps
DT = TOTAL_TIME / NUM_STEPS

# Market Parameters (Black-Scholes World)
RISK_FREE_RATE = 0.05
VOLATILITY = 0.2
STRIKE_PRICE = 100.0
INIT_PRICE = 100.0   # S0

# --- 1. THE NEURAL NETWORK ( The "Hedge Strategy" ) ---
class HedgingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: Dimension + 1 (for time)
        # Output: Dimension (Hedge for each asset)
        self.net = nn.Sequential(
            nn.Linear(DIMENSION + 1, 32),
            nn.Tanh(), # Tanh is standard for financial PDEs
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, DIMENSION) 
        )

    def forward(self, t, x):
        # We concat time t to the state x
        # t shape: [Batch, 1], x shape: [Batch, Dim]
        inputs = torch.cat([t, x], dim=1) 
        return self.net(inputs)

# --- 2. THE NEURAL SDE SOLVER ---
class DeepBSDESolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.hedge_net = HedgingNetwork()
        
        # Y0 is the Price at time 0. 
        # Crucially, we make this a LEARNABLE PARAMETER.
        self.y0 = nn.Parameter(torch.tensor([INIT_PRICE * 0.5]))
        
    def forward(self, x0, W):
        batch_size = x0.size(0)
        x = x0 # Asset Price
        y = self.y0 * torch.ones(batch_size, 1) # Portfolio Value
        
        for i in range(NUM_STEPS):
            t_val = (i * DT) * torch.ones(batch_size, 1)
            dW = W[:, i, :]
            
            # Ask Neural Net for the Hedge (Z)
            z = self.hedge_net(t_val, x)
            
            # Simple geometric brownian motion for X (Asset)
            x_next = x * (1 + RISK_FREE_RATE * DT + VOLATILITY * dW)
            
            # Update Y (BSDE dynamics)
            # y_next = y - r*y*dt + z*dW
            rhs = -RISK_FREE_RATE * y
            y_next = y + (rhs * DT) + torch.sum(z * dW, dim=1, keepdim=True)
            
            x = x_next
            y = y_next
            
        return y, x

# --- 3. TRAINING LOOP ---
def train():
    print(f"--- Training Neural SDE for {DIMENSION}D Call Option ---")
    model = DeepBSDESolver()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    loss_history = []
    start_time = time.time()
    
    for epoch in range(500):
        optimizer.zero_grad()
        
        # 1. Generate Brownian Motion
        W = torch.randn(BATCH_SIZE, NUM_STEPS, DIMENSION) * np.sqrt(DT)
        
        # 2. Initial Asset State
        x0 = torch.ones(BATCH_SIZE, DIMENSION) * INIT_PRICE
        
        # 3. Run Forward SDE
        y_terminal_pred, x_terminal = model(x0, W)
        
        # 4. Calculate ACTUAL Payoff (Target)
        s_T = torch.mean(x_terminal, dim=1, keepdim=True)
        payoff = torch.relu(s_T - STRIKE_PRICE)
        
        # 5. Loss
        loss = torch.mean((y_terminal_pred - payoff)**2)
        
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}, Price (Y0): {model.y0.item():.4f}")

    print(f"--- Done in {time.time() - start_time:.2f}s ---")
    
    # Analytical Check
    d1 = (np.log(INIT_PRICE/STRIKE_PRICE) + (RISK_FREE_RATE + 0.5 * VOLATILITY**2) * TOTAL_TIME) / (VOLATILITY * np.sqrt(TOTAL_TIME))
    d2 = d1 - VOLATILITY * np.sqrt(TOTAL_TIME)
    bs_price = INIT_PRICE * norm.cdf(d1) - STRIKE_PRICE * np.exp(-RISK_FREE_RATE * TOTAL_TIME) * norm.cdf(d2)
    
    print(f"\nExact Black-Scholes Price: {bs_price:.4f}")
    print(f"Neural SDE Learned Price:  {model.y0.item():.4f}")

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(loss_history)
    plt.title("Neural SDE Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
"""

QUANTLIB_KVA_SOURCE = r"""import QuantLib as ql
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
"""

# --- UTILITY: Create Notebook JSON ---
def create_notebook_json(code_source):
    """Wraps python code in a Jupyter Notebook JSON structure."""
    return json.dumps({
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Interactive Demo\n",
                    "Run the cell below to execute the simulation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code_source.splitlines(True)
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }, indent=2)

# --- MAIN GENERATION SCRIPT ---

def create_repo():
    # 1. Create Directories
    dirs = [
        REPO_NAME,
        os.path.join(REPO_NAME, "src"),
        os.path.join(REPO_NAME, "docs"),
        os.path.join(REPO_NAME, "notebooks")
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

    # 2. Write Files
    files = {
        os.path.join(REPO_NAME, "README.md"): README_CONTENT,
        os.path.join(REPO_NAME, "requirements.txt"): REQUIREMENTS_CONTENT,
        os.path.join(REPO_NAME, "docs", "theory.md"): THEORY_CONTENT,
        os.path.join(REPO_NAME, "src", "__init__.py"): "",
        os.path.join(REPO_NAME, "src", "neural_sde_solver.py"): NEURAL_SDE_SOURCE,
        os.path.join(REPO_NAME, "src", "quantlib_kva.py"): QUANTLIB_KVA_SOURCE,
        os.path.join(REPO_NAME, "notebooks", "01_neural_sde_demo.ipynb"): create_notebook_json(NEURAL_SDE_SOURCE),
        os.path.join(REPO_NAME, "notebooks", "02_quantlib_kva_demo.ipynb"): create_notebook_json(QUANTLIB_KVA_SOURCE),
    }

    for filepath, content in files.items():
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {filepath}")

    print("\n" + "="*50)
    print(f"Repository '{REPO_NAME}' successfully generated!")
    print("="*50)
    print("Next steps:")
    print(f"1. cd {REPO_NAME}")
    print("2. pip install -r requirements.txt")
    print("3. python src/neural_sde_solver.py")

if __name__ == "__main__":
    create_repo()