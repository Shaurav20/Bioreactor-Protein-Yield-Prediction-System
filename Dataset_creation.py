"""
Simple Bioreactor Batch Dataset Generator
Generates random batch data with different glucose feed rates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# ============================================
# GENERATE SIMPLE BATCH DATASET
# ============================================

def generate_simple_bioreactor_data():
    """
    Generate simple bioreactor dataset with 5 batches
    Each batch has different glucose feed rate
    """
    
    # Parameters
    n_batches = 5
    n_timepoints = 50  # 50 time points per batch
    time = np.arange(n_timepoints)  # 0,1,2,...,49
    
    # Different feed rates for each batch
    feed_rates = [0.8, 1.2, 1.5, 1.0, 1.8]  # g/L/hr
    
    all_data = []
    
    print("Generating bioreactor batch data...")
    print("-" * 50)
    
    for batch_id in range(n_batches):
        
        # Current batch feed rate
        feed_rate = feed_rates[batch_id]
        
        # 1. Glucose feed rate (constant for this batch)
        glucose_feed = np.ones(n_timepoints) * feed_rate
        
        # 2. Process parameters (with random noise)
        temperature = 37 + np.random.normal(0, 0.3, n_timepoints)
        ph = 7.0 + np.random.normal(0, 0.1, n_timepoints)
        do = 70 + np.random.normal(0, 5, n_timepoints)
        
        # 3. Protein yield (affected by feed rate and time)
        # Higher feed rate = higher yield, but with random variation
        base_yield = 0.1 * time  # Linear increase
        feed_effect = feed_rate * 0.5
        noise = np.random.normal(0, 0.2, n_timepoints)
        
        protein_yield = base_yield + feed_effect + noise
        protein_yield = np.maximum(protein_yield, 0)  # No negative yields
        
        # Create batch dataframe
        batch_df = pd.DataFrame({
            'Batch_ID': f'Batch_{batch_id+1}',
            'Feed_Rate_gLh': round(feed_rate, 2),
            'Time_h': time,
            'Temperature_C': np.round(temperature, 2),
            'pH': np.round(ph, 2),
            'DO_percent': np.round(do, 1),
            'Protein_Yield_gL': np.round(protein_yield, 3)
        })
        
        all_data.append(batch_df)
        
        print(f"Batch {batch_id+1}: Feed Rate = {feed_rate} g/L/hr, "
              f"Final Yield = {protein_yield[-1]:.2f} g/L")
    
    # Combine all batches
    df = pd.concat(all_data, ignore_index=True)
    
    print("-" * 50)
    print(f"Total dataset size: {df.shape}")
    print(f"Batches: {df['Batch_ID'].nunique()}")
    print(f"Timepoints per batch: {n_timepoints}")
    print(f"Columns: {list(df.columns)}")
    
    return df

# ============================================
# VISUALIZE
# ============================================

def plot_simple_data(df):
    """
    Simple visualization of the dataset
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Protein yield over time
    ax = axes[0, 0]
    for batch in df['Batch_ID'].unique():
        batch_data = df[df['Batch_ID'] == batch]
        ax.plot(batch_data['Time_h'], batch_data['Protein_Yield_gL'], 
                label=f"{batch} (Feed: {batch_data['Feed_Rate_gLh'].iloc[0]} g/L/hr)")
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Protein Yield (g/L)')
    ax.set_title('Protein Yield Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Final yield vs Feed rate
    ax = axes[0, 1]
    final_yields = df.groupby('Batch_ID').last()['Protein_Yield_gL']
    feed_rates = df.groupby('Batch_ID')['Feed_Rate_gLh'].first()
    ax.scatter(feed_rates, final_yields, s=100)
    for batch in feed_rates.index:
        ax.annotate(batch, (feed_rates[batch], final_yields[batch]), 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Feed Rate (g/L/hr)')
    ax.set_ylabel('Final Yield (g/L)')
    ax.set_title('Effect of Feed Rate on Final Yield')
    ax.grid(True, alpha=0.3)
    
    # 3. Temperature distribution
    ax = axes[1, 0]
    for batch in df['Batch_ID'].unique():
        batch_data = df[df['Batch_ID'] == batch]
        ax.hist(batch_data['Temperature_C'], alpha=0.5, label=batch, bins=10)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Frequency')
    ax.set_title('Temperature Distribution by Batch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. pH distribution
    ax = axes[1, 1]
    for batch in df['Batch_ID'].unique():
        batch_data = df[df['Batch_ID'] == batch]
        ax.hist(batch_data['pH'], alpha=0.5, label=batch, bins=10)
    ax.set_xlabel('pH')
    ax.set_ylabel('Frequency')
    ax.set_title('pH Distribution by Batch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_bioreactor_data.png', dpi=150)
    plt.show()
    
    print("\nPlot saved as 'simple_bioreactor_data.png'")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    
    # Generate dataset
    df = generate_simple_bioreactor_data()
    
    # Show first 10 rows
    print("\nFirst 10 rows of dataset:")
    print(df.head(10))
    
    # Basic statistics
    print("\nDataset Statistics:")
    print(df.groupby('Batch_ID')['Protein_Yield_gL'].describe())
    
    # Plot
    plot_simple_data(df)
    
    # Save to CSV
    df.to_csv('bioreactor_batch_data.csv', index=False)
    print("\nDataset saved to 'bioreactor_batch_data.csv'")