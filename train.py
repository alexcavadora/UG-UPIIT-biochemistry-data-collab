import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from code.loader import DatasetBuilder
from code.timeseries_analysis import run_time_series_analysis

if __name__ == "__main__":
    builder = DatasetBuilder()
    dataset = builder.build_full_dataset()
    print(f"Dataset loaded: {dataset.shape}")
    run_time_series_analysis(dataset)
