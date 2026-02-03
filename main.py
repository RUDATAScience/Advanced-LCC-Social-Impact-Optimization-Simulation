# ======================================================================
# Advanced LCC & Social Impact Optimization Simulation
# (Multi-Perspective Verification Suite)
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from numpy.random import Generator, PCG64
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Visualization Style
# ----------------------------------------------------------------------
def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'figure.figsize': (10, 6),
        'figure.dpi': 120
    })

# 2. Simulator Class
# ----------------------------------------------------------------------
class LCCSimulator:
    def __init__(self, seed=42):
        self.rng = Generator(PCG64(seed))
        self.states = ['A', 'B', 'C', 'D']
        
        # Default Cost Parameters (Can be overridden)
        self.default_costs = {
            'inspection': 1,
            'repair_B': 15,
            'repair_C': 60,
            'replace_D': 250,
            'social_loss_D': 500  # Default Impact of an accident/failure
        }

    def _get_next_state(self, current_state_idx, probs):
        if self.states[current_state_idx] == 'D':
            return 3
        
        current_s = self.states[current_state_idx]
        next_s = self.states[current_state_idx + 1]
        key = f"{current_s}->{next_s}"
        p = probs.get(key, 0)
        
        if self.rng.random() < p:
            return current_state_idx + 1
        return current_state_idx

    def run_cohort_simulation(self, strategy, transition_probs, cost_params=None, 
                              duration_years=50, num_units=1000):
        """
        Simulates a cohort of units. Returns detailed array of costs for distribution analysis.
        """
        if cost_params is None:
            cost_params = self.default_costs
            
        months = duration_years * 12
        
        # Arrays to store results per unit
        unit_direct_costs = np.zeros(num_units)
        unit_social_costs = np.zeros(num_units)
        unit_failures = np.zeros(num_units)
        
        for i in range(num_units):
            curr_state = 0
            d_cost = 0
            s_cost = 0
            fails = 0
            
            for m in range(months):
                d_cost += cost_params['inspection']
                action_taken = False
                
                # Maintenance Logic
                if strategy == 'preventive_B': # Pre-emptive
                    if curr_state >= 1:
                        if curr_state == 1: cost = cost_params['repair_B']
                        elif curr_state == 2: cost = cost_params['repair_C']
                        else: cost = cost_params['replace_D'] + cost_params['social_loss_D']
                        
                        d_cost += cost if curr_state != 3 else cost_params['replace_D']
                        if curr_state == 3:
                            s_cost += cost_params['social_loss_D']
                            fails += 1
                        curr_state = 0
                        action_taken = True

                elif strategy == 'preventive_C': # Preventive
                    if curr_state >= 2:
                        if curr_state == 2: cost = cost_params['repair_C']
                        else: cost = cost_params['replace_D'] + cost_params['social_loss_D']
                        
                        d_cost += cost if curr_state != 3 else cost_params['replace_D']
                        if curr_state == 3:
                            s_cost += cost_params['social_loss_D']
                            fails += 1
                        curr_state = 0
                        action_taken = True
                
                elif strategy == 'corrective': # Run-to-Failure
                    if curr_state == 3:
                        d_cost += cost_params['replace_D']
                        s_cost += cost_params['social_loss_D']
                        fails += 1
                        curr_state = 0
                        action_taken = True
                
                # Transition
                if not action_taken:
                    curr_state = self._get_next_state(curr_state, transition_probs)
                else:
                    curr_state = self._get_next_state(0, transition_probs)

            unit_direct_costs[i] = d_cost
            unit_social_costs[i] = s_cost
            unit_failures[i] = fails

        return {
            'strategy': strategy,
            'direct_costs': unit_direct_costs,
            'social_costs': unit_social_costs,
            'total_costs': unit_direct_costs + unit_social_costs,
            'failures': unit_failures
        }

# 3. Experiment Managers (Multi-Perspective)
# ----------------------------------------------------------------------

def run_comprehensive_analysis():
    set_publication_style()
    sim = LCCSimulator()
    base_probs = {'A->B': 0.010, 'B->C': 0.020, 'C->D': 0.030}
    strategies = ['corrective', 'preventive_C', 'preventive_B']
    labels = {
        'corrective': 'Corrective (State D)',
        'preventive_C': 'Preventive (State C)',
        'preventive_B': 'Pre-emptive (State B)'
    }
    colors = {'corrective': '#c44e52', 'preventive_C': '#dd8452', 'preventive_B': '#4c72b0'}

    # ==================================================================
    # Perspective 1: Convergence Analysis (Accuracy Verification)
    # ==================================================================
    print("--- Perspective 1: Convergence Check ---")
    n_steps = [100, 500, 1000, 5000, 10000] # Log scale steps
    conv_results = []
    
    for n in tqdm(n_steps):
        for strat in strategies:
            res = sim.run_cohort_simulation(strat, base_probs, num_units=n)
            mean_cost = np.mean(res['total_costs'])
            std_error = np.std(res['total_costs']) / np.sqrt(n)
            conv_results.append({'N': n, 'Strategy': labels[strat], 'MeanCost': mean_cost, 'SE': std_error})
    
    df_conv = pd.DataFrame(conv_results)
    
    plt.figure(figsize=(10, 5))
    for strat in df_conv['Strategy'].unique():
        subset = df_conv[df_conv['Strategy'] == strat]
        plt.errorbar(subset['N'], subset['MeanCost'], yerr=subset['SE']*1.96, 
                     marker='o', label=strat, capsize=5)
    plt.xscale('log')
    plt.xlabel('Number of Simulation Trials (N)')
    plt.ylabel('Mean Total Cost (with 95% CI)')
    plt.title('Perspective 1: Convergence of Calculation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==================================================================
    # Perspective 2: Risk Profile (Distribution Analysis)
    # ==================================================================
    print("\n--- Perspective 2: Risk Profile Analysis (N=10,000) ---")
    dist_data = []
    fixed_n = 10000
    
    plt.figure(figsize=(10, 6))
    for strat in strategies:
        res = sim.run_cohort_simulation(strat, base_probs, num_units=fixed_n)
        sns.kdeplot(res['total_costs'], label=labels[strat], color=colors[strat], fill=True, alpha=0.3)
        dist_data.append(res)
        
    plt.xlabel('Total Lifecycle Cost per Unit (Direct + Social)')
    plt.ylabel('Probability Density')
    plt.title('Perspective 2: Risk Profile Distribution (Fat Tail Analysis)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==================================================================
    # Perspective 3: Sensitivity Analysis (Social Cost Impact)
    # ==================================================================
    print("\n--- Perspective 3: Sensitivity to Social Cost Parameter ---")
    social_costs_range = np.linspace(0, 1000, 11) # Vary social cost from 0 to 1000
    sens_results = []
    
    for sc in tqdm(social_costs_range):
        current_params = sim.default_costs.copy()
        current_params['social_loss_D'] = sc
        for strat in strategies:
            res = sim.run_cohort_simulation(strat, base_probs, cost_params=current_params, num_units=2000)
            sens_results.append({
                'SocialCostParam': sc,
                'Strategy': labels[strat],
                'MeanTotalCost': np.mean(res['total_costs'])
            })
            
    df_sens = pd.DataFrame(sens_results)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_sens, x='SocialCostParam', y='MeanTotalCost', hue='Strategy', marker='o', palette=[colors[s] for s in strategies])
    plt.axvline(x=500, color='gray', linestyle=':', label='Baseline Assumption (500)')
    plt.xlabel('Assumed Social Cost of Failure (Unit Price)')
    plt.ylabel('Mean Total Lifecycle Cost')
    plt.title('Perspective 3: Sensitivity Analysis (tipping Point Identification)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # ==================================================================
    # Perspective 4: Cost-Risk Trade-off (The "Sweet Spot")
    # ==================================================================
    print("\n--- Perspective 4: Direct Cost vs Social Risk Trade-off ---")
    # Use the data from the Baseline run (N=10000)
    tradeoff_data = []
    for i, strat in enumerate(strategies):
        res = dist_data[i] # Retrieved from Perspective 2 loop
        mean_direct = np.mean(res['direct_costs'])
        mean_failures = np.mean(res['failures'])
        tradeoff_data.append({
            'Strategy': labels[strat],
            'DirectCost': mean_direct,
            'Risk (Failures)': mean_failures
        })
    
    df_tradeoff = pd.DataFrame(tradeoff_data)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_tradeoff, x='Risk (Failures)', y='DirectCost', hue='Strategy', s=300, palette=[colors[s] for s in strategies])
    
    # Annotate
    for i in range(len(df_tradeoff)):
        plt.text(df_tradeoff['Risk (Failures)'][i], df_tradeoff['DirectCost'][i]+10, 
                 df_tradeoff['Strategy'][i], ha='center')
        
    plt.title('Perspective 4: Optimization Landscape (Cost vs Risk)')
    plt.xlabel('Social Risk (Avg Failures per Unit)')
    plt.ylabel('Direct Maintenance Cost')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comprehensive_analysis()