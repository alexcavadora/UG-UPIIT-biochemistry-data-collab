import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from code.loader import DatasetBuilder
from code.ml import run_baseline_pipeline

if __name__ == "__main__":
    print("Loading dataset...")
    builder = DatasetBuilder()
    dataset = builder.build_full_dataset()
    print(f"Dataset loaded: {dataset.shape}")

    print("\nPreparing visualization and ML pipeline...")
    pipeline = run_baseline_pipeline(dataset)

    pipeline.summary()
    print("\nPipeline complete. Check results/ and models/ directories.")
