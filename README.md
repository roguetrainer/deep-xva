# **deep-xva**

**PyTorch Neural SDE solver for fast XVA/KVA. Solves high-dimensional BSDEs using deep learning to calculate capital profiles without nested simulations.**

## **The XVA Context: Why this exists**

For a detailed history of how the financial industry moved from theoretical arguments to the current consensus, please read [**The Evolution of XVA**](./docs/XVA-DEBATES.md) included in this repository.

---
![XVA](./img/Neural-XVA.png)
---

### **The Hull & White Connection**

This repository is inspired by the intellectual journey of **John Hull and Alan White**.

1. **The Critics (2010s):** Originally, Hull & White argued that Funding Valuation Adjustments (FVA) were theoretically incorrect.  
2. **The Pivot (2020s):** As the market ignored theory and adopted FVA/KVA anyway, Hull & White recognized a new, more urgent problem: **Computation**.

Calculating these adjustments (especially KVA for Basel III) requires **Nested Monte Carlo Simulations** (simulating capital requirements *inside* a market simulation). This is computationally prohibitive ($10^7+$ calculations per trade).

Consequently, Hull & White (and others like Cornelis Oosterlee) pivoted to **Deep Learning**. They proposed using Neural Networks to approximate the hedging and capital functions, collapsing the "nested" loop into a single forward pass.

**This repository implements that Neural SDE approach.**

## **Overview**

This repository demonstrates two approaches to calculating Valuation Adjustments:

1. **The "Classic" Approach (src/quantlib\_kva.py):** \* Uses QuantLib and a Hull-White model.  
   * Demonstrates the computational bottleneck: to calculate KVA, we must use a "proxy" pricing formula because running a nested simulation is too slow for a simple demo.  
2. **The "Deep Learning" Approach (src/neural\_sde\_solver.py):** \* Uses PyTorch to implement a **Deep BSDE (Backward Stochastic Differential Equation)** solver.  
   * **How it works:** Instead of simulating the inner loop, a Neural Network learns the gradient (hedging strategy) required to replicate the portfolio.  
   * **Result:** It calculates the entire capital profile in milliseconds.

## **Installation**

1. Create a virtual environment:  
   python \-m venv venv  
   source venv/bin/activate  \# or venv\\Scripts\\activate on Windows

2. Install dependencies:  
   pip install \-r requirements.txt

## **Usage**

### **Run as Scripts**

You can run the pure Python scripts directly:

\# Run the Neural SDE Solver (Fast, Modern)  
python src/neural\_sde\_solver.py

\# Run the QuantLib KVA Calculator (Slow, Classic)  
python src/quantlib\_kva.py

### **Run Notebooks**

Navigate to the notebooks/ directory and launch Jupyter to visualize the convergence of the Neural SDE:

jupyter notebook  