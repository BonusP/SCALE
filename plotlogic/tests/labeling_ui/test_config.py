from labeling_ui.config import get_dataset_root

def test_dataset_root_exists():
    root = get_dataset_root()
    assert root.endswith("data/datasets/phase1_samples")
