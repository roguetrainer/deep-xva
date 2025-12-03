# Neural SDEs and the KVA Problem

## The Problem: KVA (Capital Valuation Adjustment)

KVA is the cost of holding regulatory capital against a derivative trade for its entire lifetime.

To calculate it accurately, you need a **Nested Simulation**:
1.  **Outer Loop:** Simulate the market forward (e.g., 1,000 paths for 30 years).
2.  **Inner Loop:** At *every* time step and *every* path of the Outer Loop, you must calculate the "Regulatory Capital."
    * Under Basel III, this often requires calculating "Expected Shortfall," which requires *another* simulation (e.g., 1,000 paths) starting from that future point.

**Total Complexity:** $1,000 \times 30 \text{ years} \times 1,000 = 30,000,000$ valuations. This is computationally prohibitive.

## The Solution: Neural SDEs (Deep BSDE)

A Neural SDE (Stochastic Differential Equation) replaces the "Inner Loop" with a Neural Network.

We treat the **Capital Requirement** (or the Option Price) and the **Hedging Strategy** as unknown functions to be learned.

1.  We define a Neural Network $Z(t, X_t)$ that takes the current time and market state and predicts the Hedge/Gradient.
2.  We run *only* the Outer Loop.
3.  We use the output of the Neural Network to step the portfolio value forward in time.
4.  **Loss Function:** We force the final portfolio value to equal the Regulatory Payoff. Backpropagation updates the Neural Network to correct the hedge.

**Result:** The Neural Network "learns" the entire capital surface. Once trained, getting the capital requirement at any future time takes milliseconds (one inference) rather than hours (nested Monte Carlo).
