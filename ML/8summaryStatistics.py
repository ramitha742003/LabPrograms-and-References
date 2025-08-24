# Write a program to compute summary statistics such as mean, median, mode, standard deviation & variance of given different types of data.

import statistics
from collections import Counter

def calculate_summary_statistics(data):
    """
    Computes and prints summary statistics
    for a given list of numerical data.

    Args:
        data (list): A list of numerical values
    """
    if not data:
        print("The provided data list is empty. Cannot compute statistics.")
        return

    # Calculate mean and median
    mean_value = statistics.mean(data)
    median_value = statistics.median(data)

    # Attempt to compute mode using statistics module
    try:
        mode_value = statistics.mode(data)
    except statistics.StatisticsError:
        counts = Counter(data)
        max_count = 0
        modes = []
        # Find highest frequency
        for value, count in counts.items():
            if count > max_count:
                max_count = count
                modes = [value]
            elif count == max_count and max_count > 1:
                modes.append(value)
        # Assign result to mode_value
        if modes and max_count > 1:
            mode_value = modes
        else:
            mode_value = "No unique mode (or all values are unique)"

    # Calculate variance and standard deviation
    variance_value = statistics.variance(data)
    std_dev_value = statistics.stdev(data)

    # Print all results
    print(f"Dataset: {data}")
    print(f"Mean: {mean_value}")
    print(f"Median: {median_value}")
    print(f"Mode: {mode_value}")
    print(f"Variance: {variance_value}")
    print(f"Standard Deviation: {std_dev_value}")

# ----------------- Test Cases -----------------

# Integer dataset
print("\n--- Integers Data ---")
integers_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
calculate_summary_statistics(integers_data)

# Float dataset
print("\n--- Float Data ---")
float_data = [1.5, 2.3, 3.1, 4.8, 5.0, 5.0, 6.2, 7.9]
calculate_summary_statistics(float_data)

# Data with multiple modes
print("\n--- Data with Multiple Modes ---")
multimode_data = [1, 2, 2, 3, 4, 4, 5]
calculate_summary_statistics(multimode_data)

# Data with all unique values (no mode)
print("\n--- Data with No Mode (All Unique) ---")
no_mode_data = [10, 20, 30, 40, 50]
calculate_summary_statistics(no_mode_data)

# Empty dataset
print("\n--- Empty Data ---")
empty_data = []
calculate_summary_statistics(empty_data)
