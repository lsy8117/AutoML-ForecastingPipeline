import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10):
    """Compute PSI for a single feature."""
    bin_edges = np.histogram_bin_edges(expected, bins=bins)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = np.clip(expected_counts / expected_counts.sum(), 1e-10, 1)
    actual_percents = np.clip(actual_counts / actual_counts.sum(), 1e-10, 1)

    psi_values = (expected_percents - actual_percents) * np.log(expected_percents / actual_percents)
    return np.sum(psi_values)


# Simulated Training and Test Data (Multiple Features)
np.random.seed(42)
train_data = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, 1000),
    'feature_2': np.random.normal(5, 2, 1000),
    'feature_3': np.random.uniform(0, 1, 1000)
})

test_data = pd.DataFrame({
    'feature_1': np.random.normal(0.2, 1, 800),  # Shifted
    'feature_2': np.random.normal(5.5, 2, 800),  # Shifted
    'feature_3': np.random.uniform(0, 1, 800)  # No shift
})

# Compute PSI for all features
psi_results = {col: calculate_psi(train_data[col], test_data[col]) for col in train_data.columns}
psi_df = pd.DataFrame(list(psi_results.items()), columns=['Feature', 'PSI Value'])

# Display Results
print(psi_df)
