# ======================================================================
# Advanced LCC & Social Impact Optimization Simulation
# (Multi-Perspective Verification Suite: Enhanced Statistical Rigor & CSV Export)
# ======================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from numpy.random import Generator, PCG64
from scipy import stats  # Added for statistical testing
import warnings
import os

# Google Colab specific import (wrapped in try-except for safety)
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print("Warning: Not running in Google Colab. Files will be saved locally but not auto-downloaded.")

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

# 2. Simulator Class (Vectorized for Speed)
# ----------------------------------------------------------------------
class LCCSimulator:
    def __init__(self, seed=42):
        self.rng = Generator(PCG64(seed))
        self.default_costs = {
            'inspection': 1,
            'repair_B': 15,
            'repair_C': 60,
            'replace_D': 250,
            'social_loss_D': 500
        }

    def run_cohort_simulation(self, strategy, transition_probs, cost_params=None,
                              duration_years=50, num_units=1000):
        if cost_params is None: cost_params = self.default_costs
        months = duration_years * 12

        # State: 0=A, 1=B, 2=C, 3=D
        current_states = np.zeros(num_units, dtype=int)

        unit_direct_costs = np.zeros(num_units)
        unit_social_costs = np.zeros(num_units)
        unit_failures = np.zeros(num_units)

        p_ab = transition_probs.get('A->B', 0)
        p_bc = transition_probs.get('B->C', 0)
        p_cd = transition_probs.get('C->D', 0)

        for _ in range(months):
            unit_direct_costs += cost_params['inspection']
            action_mask = np.zeros(num_units, dtype=bool)

            if strategy == 'preventive_B':
                mask_b = (current_states == 1); unit_direct_costs[mask_b] += cost_params['repair_B']
                mask_c = (current_states == 2); unit_direct_costs[mask_c] += cost_params['repair_C']
                mask_d = (current_states == 3); unit_direct_costs[mask_d] += cost_params['replace_D']
                unit_social_costs[mask_d] += cost_params['social_loss_D']; unit_failures[mask_d] += 1
                action_mask = mask_b | mask_c | mask_d

            elif strategy == 'preventive_C':
                mask_c = (current_states == 2); unit_direct_costs[mask_c] += cost_params['repair_C']
                mask_d = (current_states == 3); unit_direct_costs[mask_d] += cost_params['replace_D']
                unit_social_costs[mask_d] += cost_params['social_loss_D']; unit_failures[mask_d] += 1
                action_mask = mask_c | mask_d

            elif strategy == 'corrective':
                mask_d = (current_states == 3); unit_direct_costs[mask_d] += cost_params['replace_D']
                unit_social_costs[mask_d] += cost_params['social_loss_D']; unit_failures[mask_d] += 1
                action_mask = mask_d

            # Apply Maintenance
            current_states[action_mask] = 0

            # Transitions
            rand_vals = self.rng.random(num_units)
            mask_0 = (current_states == 0); trans_0 = mask_0 & (rand_vals < p_ab)
            mask_1 = (current_states == 1); trans_1 = mask_1 & (rand_vals < p_bc)
            mask_2 = (current_states == 2); trans_2 = mask_2 & (rand_vals < p_cd)

            current_states[trans_0] = 1
            current_states[trans_1] = 2
            current_states[trans_2] = 3

        return {
            'strategy': strategy,
            'total_costs': unit_direct_costs + unit_social_costs,
            'direct_costs': unit_direct_costs,
            'failures': unit_failures
        }

# 3. Comprehensive Analysis
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

    # Dictionary to store all dataframes for export
    export_dfs = {}

    # ==================================================================
    # Perspective 1: Convergence Analysis
    # ==================================================================
    print("--- Perspective 1: Convergence Check ---")
    n_steps = [1000, 10000, 100000]
    conv_results = []

    for n in tqdm(n_steps, desc="Convergence Loop"):
        for strat in strategies:
            res = sim.run_cohort_simulation(strat, base_probs, num_units=n)
            mean = np.mean(res['total_costs'])
            se = np.std(res['total_costs']) / np.sqrt(n)
            conv_results.append({'N': n, 'Strategy': labels[strat], 'MeanCost': mean, 'SE': se})

    df_conv = pd.DataFrame(conv_results)
    export_dfs['Perspective1_Convergence'] = df_conv # Store for export

    plt.figure(figsize=(10, 5))
    for strat in df_conv['Strategy'].unique():
        sub = df_conv[df_conv['Strategy'] == strat]
        plt.errorbar(sub['N'], sub['MeanCost'], yerr=sub['SE']*1.96, marker='o', label=strat, capsize=5)
    plt.xscale('log'); plt.xlabel('N (Trials)'); plt.ylabel('Mean Cost'); plt.title('Perspective 1: Convergence of Calculation Accuracy'); plt.legend(); plt.show()

    # ==================================================================
    # Perspective 2: Risk Profile Evolution (1k -> 100k)
    # ==================================================================
    print("\n--- Perspective 2: Risk Profile Evolution ---")
    final_results = {}

    for n in n_steps:
        print(f"Plotting Distribution for N={n}...")
        plt.figure(figsize=(10, 6))
        for strat in strategies:
            res = sim.run_cohort_simulation(strat, base_probs, num_units=n)
            sns.kdeplot(res['total_costs'], label=labels[strat], color=colors[strat], fill=True, alpha=0.3)
            if n == 100000: final_results[strat] = res['total_costs'] # Store for stats
        plt.xlabel('Total Cost'); plt.ylabel('Density'); plt.title(f'Perspective 2: Risk Profile (N={n})'); plt.legend(); plt.show()

    # Create a DataFrame for N=100,000 distribution data
    # Be careful: DataFrames require equal length, which is true here (N=100k)
    df_dist_100k = pd.DataFrame(final_results)
    # Rename columns to friendly labels if keys are raw strategy names
    df_dist_100k.columns = [labels[k] for k in final_results.keys()]
    export_dfs['Perspective2_Distribution_N100k'] = df_dist_100k

    # ==================================================================
    # Perspective 3: Sensitivity Analysis
    # ==================================================================
    print("\n--- Perspective 3: Sensitivity Analysis ---")
    sens_results = []
    for sc in tqdm(np.linspace(0, 1000, 11), desc="Sensitivity Loop"):
        cp = sim.default_costs.copy(); cp['social_loss_D'] = sc
        for strat in strategies:
            res = sim.run_cohort_simulation(strat, base_probs, cost_params=cp, num_units=10000)
            sens_results.append({'SocialCost': sc, 'Strategy': labels[strat], 'MeanCost': np.mean(res['total_costs'])})

    df_sens = pd.DataFrame(sens_results)
    export_dfs['Perspective3_Sensitivity'] = df_sens # Store for export

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_sens, x='SocialCost', y='MeanCost', hue='Strategy', marker='o', palette=[colors[s] for s in strategies])
    plt.axvline(500, color='gray', linestyle=':', label='Baseline'); plt.title('Perspective 3: Sensitivity Analysis'); plt.legend(); plt.show()

    # ==================================================================
    # Perspective 4: Cost-Risk Trade-off
    # ==================================================================
    print("\n--- Perspective 4: Cost-Risk Trade-off ---")
    tradeoff_data = []
    for strat in strategies:
        # Re-run for clean separation of metrics, though we could reuse final_results if structured differently
        res = sim.run_cohort_simulation(strat, base_probs, num_units=100000)
        tradeoff_data.append({'Strategy': labels[strat], 'Direct': np.mean(res['direct_costs']), 'Risk': np.mean(res['failures'])})

    df_to = pd.DataFrame(tradeoff_data)
    export_dfs['Perspective4_Tradeoff'] = df_to # Store for export

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_to, x='Risk', y='Direct', hue='Strategy', s=300, palette=[colors[s] for s in strategies])
    for i in range(len(df_to)): plt.text(df_to['Risk'][i], df_to['Direct'][i]+10, df_to['Strategy'][i], ha='center')
    plt.title('Perspective 4: Optimization Landscape'); plt.grid(True); plt.show()

    # ==================================================================
    # Perspective 5: Statistical Significance Testing
    # ==================================================================
    print("\n--- Perspective 5: Statistical Significance Testing (N=100,000) ---")
    corr_costs = final_results['corrective']
    prev_costs = final_results['preventive_C']

    # Welch's t-test
    t_stat, p_val_t = stats.ttest_ind(corr_costs, prev_costs, equal_var=False)
    # Mann-Whitney U test
    u_stat, p_val_u = stats.mannwhitneyu(corr_costs, prev_costs, alternative='two-sided')

    print(f"Comparison: Corrective vs. Preventive (State C)")
    print(f"  > Welch's t-test: t={t_stat:.2f}, p-value={p_val_t:.3e}")
    print(f"  > Mann-Whitney U: U={u_stat:.2e}, p-value={p_val_u:.3e}")
    if p_val_t < 0.05: print("  => Result: Statistically Significant Difference (p < 0.05)")

    # Store Stats Results
    stats_data = [{
        'Test': "Welch's t-test",
        'Statistic': t_stat,
        'P-Value': p_val_t,
        'Significant': p_val_t < 0.05
    }, {
        'Test': "Mann-Whitney U",
        'Statistic': u_stat,
        'P-Value': p_val_u,
        'Significant': p_val_u < 0.05
    }]
    export_dfs['Perspective5_StatsTests'] = pd.DataFrame(stats_data)

    # ==================================================================
    # Perspective 6: Tail Risk Quantification (VaR / CVaR)
    # ==================================================================
    print("\n--- Perspective 6: Tail Risk Quantification (VaR / CVaR) ---")
    risk_metrics = []
    for strat in strategies:
        data = final_results[strat]
        var_95 = np.percentile(data, 95)
        var_99 = np.percentile(data, 99)
        cvar_95 = data[data >= var_95].mean()
        risk_metrics.append({
            'Strategy': labels[strat],
            'VaR (95%)': var_95,
            'VaR (99%)': var_99,
            'CVaR (95%)': cvar_95
        })

    df_risk = pd.DataFrame(risk_metrics)
    export_dfs['Perspective6_RiskMetrics'] = df_risk # Store for export
    print(df_risk.to_string(index=False))

    # Bar Chart for Risk Metrics
    df_risk.set_index('Strategy')[['VaR (95%)', 'CVaR (95%)']].plot(kind='bar', figsize=(10, 6), color=['#e74c3c', '#8e44ad'])
    plt.title('Perspective 6: Extreme Tail Risk Metrics (VaR & CVaR @ 95%)')
    plt.ylabel('Cost Value (Tail Loss)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # ==================================================================
    # CSV Export Logic
    # ==================================================================
    print("\n--- Exporting Data to CSV ---")
    for name, df in export_dfs.items():
        filename = f"{name}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
        if IN_COLAB:
            files.download(filename)

if __name__ == "__main__":
    run_comprehensive_analysis()