"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_image_7band():
    """Create sample 7-band satellite image."""
    import numpy as np

    return np.random.rand(256, 256, 7).astype(np.float32)


@pytest.fixture
def sample_image_12band():
    """Create sample 12-band satellite image."""
    import numpy as np

    return np.random.rand(256, 256, 12).astype(np.float32)


@pytest.fixture
def sample_segmentation_mask():
    """Create sample segmentation mask."""
    import numpy as np

    mask = np.zeros((256, 256), dtype=np.uint8)
    # Add some fire regions
    mask[50:100, 50:100] = 1
    mask[150:180, 150:180] = 2
    return mask


@pytest.fixture
def sample_binary_mask():
    """Create sample binary mask."""
    import numpy as np

    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:100, 50:100] = 1
    return mask
