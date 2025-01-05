import pandas as pd

def export_to_csv(data, filename):
    """Exports data to a CSV file."""
    data.to_csv(filename, index=False)
    print(f"Data exported successfully to {filename}")
