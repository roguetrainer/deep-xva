import torch
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
