# Stochastic LCC & Social Impact Optimization Simulation

This repository contains the simulation code for the research paper:  
**"Optimizing Infrastructure Maintenance to Minimize Social Costs: A Stochastic Approach Considering Public Health and Economic Risks"** (Tentative Title).

This framework simulates the degradation of tunnel lighting facilities using a stochastic Markov process to evaluate the trade-offs between **Direct Maintenance Costs** and **Social Costs** (e.g., accident risks, traffic congestion, and public health impacts).

## 📌 Overview

Traditional Life Cycle Cost (LCC) analysis often focuses solely on budgetary constraints. This simulation extends the LCC framework to include **Social Impact Assessment**, quantifying the hidden costs of infrastructure failure.

The simulation verifies the effectiveness of "Preventive Maintenance" over "Corrective Maintenance" through four rigorous analytical perspectives:

1.  **Convergence Check:** Verifying statistical reliability via Monte Carlo simulations ($N=10,000$).
2.  **Risk Profile Analysis:** Visualizing the "fat tail" risks of social losses associated with corrective strategies.
3.  **Sensitivity Analysis:** Identifying the "tipping point" where social costs justify preventive interventions.
4.  **Cost-Risk Optimization:** Plotting the trade-off landscape to identify the optimal intervention threshold.

## 🛠 Requirements

* Python 3.8+
* See `requirements.txt` for dependencies.

## 📦 Installation

```bash
git clone [https://github.com/YourUsername/lcc-social-impact-sim.git](https://github.com/YourUsername/lcc-social-impact-sim.git)
cd lcc-social-impact-sim
pip install -r requirements.txt