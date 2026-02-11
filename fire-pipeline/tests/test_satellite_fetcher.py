"""Unit tests for satellite data fetcher."""

from datetime import datetime

import numpy as np
import pytest

from satellite_fetcher import (
    SatelliteImage,
    MockSatelliteFetcher,
    get_fetcher,
)


class TestSatelliteImage:
    """Tests for SatelliteImage dataclass."""

    def test_properties(self):
        """Test computed properties."""
        image = SatelliteImage(
            data=np.random.rand(512, 256, 7).astype(np.float32),
            bounds=(-122.5, 37.5, -122.0, 38.0),
            crs="EPSG:4326",
            datetime=datetime.now(),
            scene_id="TEST_001",
            cloud_cover=5.0,
            platform="Test",
        )

        assert image.width == 256
        assert image.height == 512
        assert image.center == (-122.25, 37.75)


class TestMockSatelliteFetcher:
    """Tests for MockSatelliteFetcher."""

    @pytest.fixture
    def fetcher(self):
        """Create mock fetcher."""
        return MockSatelliteFetcher()

    def test_search_scenes(self, fetcher):
        """Test searching for scenes."""
        bbox = (-122.5, 37.5, -122.0, 38.0)
        date_range = ("2024-01-01", "2024-01-31")

        scenes = fetcher.search_scenes(bbox, date_range, max_cloud_cover=20)

        assert len(scenes) > 0
        assert all(s["cloud_cover"] <= 20 for s in scenes)
        assert all("id" in s for s in scenes)
        assert all("datetime" in s for s in scenes)

    def test_search_scenes_sorted_by_cloud(self, fetcher):
        """Test that scenes are sorted by cloud cover."""
        bbox = (-122.5, 37.5, -122.0, 38.0)
        date_range = ("2024-01-01", "2024-01-31")

        scenes = fetcher.search_scenes(bbox, date_range, limit=10)

        cloud_covers = [s["cloud_cover"] for s in scenes]
        assert cloud_covers == sorted(cloud_covers)

    def test_search_scenes_limit(self, fetcher):
        """Test scene limit."""
        bbox = (-122.5, 37.5, -122.0, 38.0)
        date_range = ("2024-01-01", "2024-12-31")  # Full year

        scenes = fetcher.search_scenes(bbox, date_range, limit=5)

        assert len(scenes) <= 5

    def test_fetch_scene(self, fetcher):
        """Test fetching a single scene."""
        bbox = (-122.5, 37.5, -122.0, 38.0)
        date_range = ("2024-01-01", "2024-01-31")

        scenes = fetcher.search_scenes(bbox, date_range)
        image = fetcher.fetch_scene(scenes[0], bbox=bbox)

        assert isinstance(image, SatelliteImage)
        assert image.data.shape[2] == 7  # 7 bands
        assert image.data.dtype == np.float32
        assert image.data.min() >= 0.0
        assert image.data.max() <= 1.0
        assert image.bounds == bbox

    def test_fetch_region(self, fetcher):
        """Test fetching best image for region."""
        bbox = (-122.5, 37.5, -122.0, 38.0)
        date_range = ("2024-01-01", "2024-01-31")

        image = fetcher.fetch_region(bbox, date_range, max_cloud_cover=50)

        assert image is not None
        assert isinstance(image, SatelliteImage)
        assert image.cloud_cover <= 50

    def test_mock_data_has_burned_areas(self, fetcher):
        """Test that mock data simulates burned areas."""
        bbox = (-122.5, 37.5, -122.0, 38.0)
        date_range = ("2024-01-01", "2024-01-31")

        image = fetcher.fetch_region(bbox, date_range)

        # Mock data should have varied NIR values (some low for burned)
        nir_band = image.data[:, :, 3]  # B08 (NIR)
        assert nir_band.min() < 0.5  # Some low NIR (burned simulation)

    def test_reproducible_with_scene_id(self, fetcher):
        """Test that same scene ID produces same data."""
        bbox = (-122.5, 37.5, -122.0, 38.0)
        date_range = ("2024-01-01", "2024-01-31")

        scenes = fetcher.search_scenes(bbox, date_range)
        image1 = fetcher.fetch_scene(scenes[0], bbox=bbox)
        image2 = fetcher.fetch_scene(scenes[0], bbox=bbox)

        np.testing.assert_array_equal(image1.data, image2.data)


class TestGetFetcher:
    """Tests for get_fetcher factory function."""

    def test_get_mock_fetcher(self):
        """Test getting mock fetcher."""
        fetcher = get_fetcher(use_mock=True)
        assert isinstance(fetcher, MockSatelliteFetcher)

    def test_get_real_fetcher(self):
        """Test getting real fetcher (without initializing API)."""
        from satellite_fetcher import SatelliteFetcher

        fetcher = get_fetcher(use_mock=False)
        assert isinstance(fetcher, SatelliteFetcher)
        assert fetcher._initialized is False  # Not yet connected
