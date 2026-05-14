# ======================================================================
# Advanced LCC & Social Impact Optimization Simulation
# (Part 2: Advanced Uncertainty, Stakeholder Analysis & Error Convergence)
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from numpy.random import Generator, PCG64
import os
import zipfile
import shutil
import warnings

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

warnings.filterwarnings('ignore')

def set_publication_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

class LCCSimulator:
    def __init__(self, seed=42):
        self.rng = Generator(PCG64(seed))
        self.states = ['A', 'B', 'C', 'D']
        self.default_costs = {'inspection': 1, 'repair_B': 15, 'repair_C': 60, 'replace_D': 250, 'social_loss_D': 500}

    def _get_next_state(self, current_state_idx, probs):
        if self.states[current_state_idx] == 'D': return 3
        p = probs.get(f"{self.states[current_state_idx]}->{self.states[current_state_idx + 1]}", 0)
        return current_state_idx + 1 if self.rng.random() < p else current_state_idx

    def _get_stochastic_cost(self, cost_type, base_val):
        if cost_type == 'replace':
            return max(base_val * 0.5, self.rng.normal(loc=base_val, scale=base_val * 0.15))
        elif cost_type == 'social':
            mu = np.log(base_val) - (0.8**2) / 2
            return self.rng.lognormal(mean=mu, sigma=0.8)
        return base_val

    def run_cohort_simulation(self, strategy, transition_probs, cost_params=None, duration_years=50, num_units=1000):
        if cost_params is None: cost_params = self.default_costs
        months = duration_years * 12
        unit_direct_costs, unit_social_costs, unit_failures = np.zeros(num_units), np.zeros(num_units), np.zeros(num_units)
        
        for i in range(num_units):
            curr_state, d_cost, s_cost, fails = 0, 0, 0, 0
            for m in range(months):
                d_cost += cost_params['inspection']
                action_taken = False
                
                if strategy == 'preventive_B' and curr_state >= 1:
                    cost = cost_params['repair_B'] if curr_state == 1 else cost_params['repair_C'] if curr_state == 2 else self._get_stochastic_cost('replace', cost_params['replace_D'])
                    d_cost += cost
                    if curr_state == 3:
                        s_cost += self._get_stochastic_cost('social', cost_params['social_loss_D'])
                        fails += 1
                    curr_state, action_taken = 0, True

                elif strategy == 'preventive_C' and curr_state >= 2:
                    cost = cost_params['repair_C'] if curr_state == 2 else self._get_stochastic_cost('replace', cost_params['replace_D'])
                    d_cost += cost
                    if curr_state == 3:
                        s_cost += self._get_stochastic_cost('social', cost_params['social_loss_D'])
                        fails += 1
                    curr_state, action_taken = 0, True
                
                elif strategy == 'corrective' and curr_state == 3:
                    d_cost += self._get_stochastic_cost('replace', cost_params['replace_D'])
                    s_cost += self._get_stochastic_cost('social', cost_params['social_loss_D'])
                    fails += 1
                    curr_state, action_taken = 0, True
                
                curr_state = self._get_next_state(0, transition_probs) if action_taken else self._get_next_state(curr_state, transition_probs)

            unit_direct_costs[i], unit_social_costs[i], unit_failures[i] = d_cost, s_cost, fails

        return {'strategy': strategy, 'direct_costs': unit_direct_costs, 'social_costs': unit_social_costs, 'total_costs': unit_direct_costs + unit_social_costs, 'failures': unit_failures}

def run_advanced_analyses():
    set_publication_style()
    sim = LCCSimulator()
    base_probs = {'A->B': 0.010, 'B->C': 0.020, 'C->D': 0.030}
    strategies = ['corrective', 'preventive_C', 'preventive_B']
    labels = {'corrective': 'Corrective (State D)', 'preventive_C': 'Preventive (State C)', 'preventive_B': 'Pre-emptive (State B)'}
    
    export_dir = "LCC_Advanced_Analysis"
    if os.path.exists(export_dir): shutil.rmtree(export_dir)
    os.makedirs(export_dir)

    # ==================================================================
    # Perspective 7: Stakeholder-Specific Benefit-Loss Analysis
    # ==================================================================
    print("--- Perspective 7: Stakeholder Value Breakdown ---")
    stakeholder_data = []
    for strat in strategies:
        res = sim.run_cohort_simulation(strat, base_probs, num_units=10000)
        stakeholder_data.append({
            'Strategy': labels[strat],
            'Administrator Cost (Direct)': np.mean(res['direct_costs']),
            'Social Cost (Impacts)': np.mean(res['social_costs'])
        })
    df_stakeholder = pd.DataFrame(stakeholder_data)
    df_stakeholder.to_csv(f"{export_dir}/Perspective7_StakeholderBreakdown.csv", index=False)

    df_stakeholder.set_index('Strategy').plot(kind='bar', stacked=True, color=['#3498db', '#e74c3c'], figsize=(8, 6))
    plt.title('Perspective 7: Stakeholder Value Distribution')
    plt.ylabel('Mean Lifecycle Cost')
    plt.xticks(rotation=0)
    plt.legend(['Administrator (Direct Cost)', 'Society (Social Impact)'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{export_dir}/Perspective7_StakeholderBreakdown.png")
    plt.show()

    # ==================================================================
    # Perspective 8: Robustness against Deterioration Rate Uncertainty
    # ==================================================================
    print("\n--- Perspective 8: Deterioration Rate Sensitivity ---")
    multipliers = [0.8, 1.0, 1.2] # -20%, Baseline, +20%
    robustness_data = []
    
    for mult in multipliers:
        perturbed_probs = {k: v * mult for k, v in base_probs.items()}
        for strat in strategies:
            res = sim.run_cohort_simulation(strat, perturbed_probs, num_units=5000)
            robustness_data.append({
                'Deterioration Rate': f"{int((mult-1)*100)}%" if mult != 1.0 else "Baseline",
                'Multiplier': mult,
                'Strategy': labels[strat],
                'Mean Total Cost': np.mean(res['total_costs'])
            })
            
    df_robust = pd.DataFrame(robustness_data)
    df_robust.to_csv(f"{export_dir}/Perspective8_DeteriorationRobustness.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=df_robust, x='Deterioration Rate', y='Mean Total Cost', hue='Strategy', palette=['#c44e52', '#dd8452', '#4c72b0'])
    plt.title('Perspective 8: Robustness against Engineering Uncertainty (±20% Deterioration Rate)')
    plt.ylabel('Total Lifecycle Cost')
    plt.tight_layout()
    plt.savefig(f"{export_dir}/Perspective8_DeteriorationRobustness.png")
    plt.show()

    # ==================================================================
    # Perspective 9: Monte Carlo Error Convergence (Log-Log Scale)
    # ==================================================================
    print("\n--- Perspective 9: Monte Carlo Error Analysis ---")
    n_steps = [100, 500, 1000, 5000, 10000, 50000]
    error_data = []
    
    for n in tqdm(n_steps, desc="Calculating Standard Errors"):
        res = sim.run_cohort_simulation('corrective', base_probs, num_units=n)
        se = np.std(res['total_costs']) / np.sqrt(n)
        error_data.append({'N': n, 'Standard Error (SE)': se})
        
    df_error = pd.DataFrame(error_data)
    df_error.to_csv(f"{export_dir}/Perspective9_ErrorConvergence.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(df_error['N'], df_error['Standard Error (SE)'], marker='o', linestyle='-', color='purple')
    # Theoretical 1/sqrt(N) reference line
    ref_y = df_error['Standard Error (SE)'].iloc[0] * np.sqrt(n_steps[0]) / np.sqrt(n_steps)
    plt.plot(n_steps, ref_y, linestyle='--', color='gray', label='Theoretical $1/\sqrt{N}$')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Simulation Trials (N)')
    plt.ylabel('Standard Error (Log Scale)')
    plt.title('Perspective 9: Mathematical Proof of Monte Carlo Convergence')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{export_dir}/Perspective9_ErrorConvergence.png")
    plt.show()

    # Create ZIP
    print("\n--- Creating ZIP Archive ---")
    zip_filename = "LCC_Advanced_Analysis.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files_in_dir in os.walk(export_dir):
            for file in files_in_dir:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), start=os.path.dirname(export_dir)))
                
    if IN_COLAB:
        files.download(zip_filename)
    else:
        print(f"Done! Find '{zip_filename}' in your directory.")

if __name__ == "__main__":
    run_advanced_analyses()
