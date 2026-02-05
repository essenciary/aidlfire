"""
Fire Detection Streamlit App.

Interactive app for satellite-based wildfire detection and severity mapping.

Features:
- Interactive map for region selection
- Satellite image fetching from Planetary Computer
- Real-time fire detection inference
- History of analyses with filtering
- Visualization of results

Run with:
    streamlit run app.py
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Min width for image viewer so it doesn't render as narrow strips
st.markdown(
    """<style>div[data-testid="stPlotlyChart"]{min-width:1000px !important}</style>""",
    unsafe_allow_html=True,
)

from satellite_fetcher import get_fetcher, SatelliteImage
from storage import StorageManager, AnalysisRecord
from inference import (
    FireInferencePipeline,
    InferenceResult,
    create_visualization,
    create_fire_mask_visualization,
)


# Configuration (can be set via environment variables)
STORAGE_DIR = Path(os.environ.get("FIRE_STORAGE_DIR", "./cache"))
MODEL_PATH = Path(os.environ.get("FIRE_MODEL_PATH", "./checkpoints/best_model.pt"))
USE_MOCK_FETCHER = os.environ.get("FIRE_USE_MOCK", "true").lower() in ("true", "1", "yes")


@st.cache_resource
def get_storage():
    """Get or create storage manager."""
    return StorageManager(STORAGE_DIR)


@st.cache_resource
def get_model():
    """Load the fire detection model."""
    if MODEL_PATH.exists():
        return FireInferencePipeline(MODEL_PATH)
    return None


def run_inference_mock(image: SatelliteImage) -> InferenceResult:
    """
    Run mock inference when model is not available.

    Generates synthetic results for testing the UI.
    """
    h, w = image.data.shape[:2]

    # Create mock segmentation with some "fire" areas
    np.random.seed(hash(image.scene_id) % (2**32))
    segmentation = np.zeros((h, w), dtype=np.uint8)

    # Add random fire patches
    num_fires = np.random.randint(0, 4)
    for _ in range(num_fires):
        cx = np.random.randint(50, w - 50)
        cy = np.random.randint(50, h - 50)
        radius = np.random.randint(15, 50)

        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2
        segmentation[mask] = np.random.randint(1, 5)  # Severity 1-4

    # Create mock probabilities
    probabilities = np.zeros((h, w, 5), dtype=np.float32)
    for i in range(5):
        probabilities[:, :, i] = (segmentation == i).astype(np.float32)
    probabilities += np.random.rand(h, w, 5).astype(np.float32) * 0.1
    probabilities /= probabilities.sum(axis=2, keepdims=True)

    has_fire = (segmentation > 0).any()
    fire_probs = probabilities[:, :, 1:].sum(axis=2)

    return InferenceResult(
        segmentation=segmentation,
        probabilities=probabilities,
        has_fire=has_fire,
        fire_confidence=float(fire_probs.max()) if has_fire else 0.0,
        fire_fraction=float((segmentation > 0).mean()),
        severity_counts={
            "no_damage": int((segmentation == 0).sum()),
            "negligible": int((segmentation == 1).sum()),
            "moderate": int((segmentation == 2).sum()),
            "high": int((segmentation == 3).sum()),
            "destroyed": int((segmentation == 4).sum()),
        },
        num_classes=5,
        image_shape=(h, w),
    )


def create_rgb_preview(image: SatelliteImage) -> np.ndarray:
    """Create RGB preview from satellite image."""
    # Use bands 2, 1, 0 (Red, Green, Blue)
    rgb = image.data[:, :, [2, 1, 0]]
    rgb = np.clip(rgb * 3.0, 0, 1)  # Boost brightness
    return (rgb * 255).astype(np.uint8)


def resize_for_display(img: np.ndarray, max_size: int = 600) -> np.ndarray:
    """Resize image for display, capping the longest side to max_size."""
    from PIL import Image

    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    pil = Image.fromarray(img)
    return np.array(pil.resize((new_w, new_h), Image.Resampling.LANCZOS))


def render_synced_images(
    img1: np.ndarray,
    img2: np.ndarray,
    label1: str = "Original",
    label2: str = "Fire detection",
    max_size: int = 1000,
) -> None:
    """
    Render two images side-by-side with Plotly, synced zoom/pan.
    Uses px.imshow facet_col for consistent equal sizing of both panels.
    """
    import plotly.express as px

    # Resize both to same size for display (whole image visible)
    img1 = resize_for_display(img1, max_size=max_size)
    img2 = resize_for_display(img2, max_size=max_size)

    # Ensure same dimensions for sync (crop to min)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h, w = min(h1, h2), min(w1, w2)
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]

    # Stack as (2, H, W, 3) - facet_col ensures equal sizing for both panels
    stacked = np.stack([img1, img2], axis=0)
    fig = px.imshow(
        stacked,
        facet_col=0,
        binary_string=True,
        facet_col_wrap=2,
        labels={"facet_col": ""},
    )
    # Set facet titles
    for i, label in enumerate([label1, label2]):
        if i < len(fig.layout.annotations):
            fig.layout.annotations[i].text = label

    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, scaleanchor="x")
    display_height = max(450, min(h, 600))
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        height=display_height,
        width=max(1000, w),  # Min 1000px so images don't render as narrow strips
        autosize=True,
        dragmode="pan",
        showlegend=False,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "doubleClick": "reset",
        },
    )


def main():
    """Main app function."""
    # Initialize session state for navigation
    if "page" not in st.session_state:
        st.session_state.page = "New Analysis"

    # Sidebar
    with st.sidebar:
        st.title("🔥 Fire Detection")
        st.markdown("---")

        # Navigation menu with styled buttons
        menu_items = [
            ("🛰️ New Analysis", "New Analysis"),
            ("📜 History", "History"),
            ("📊 Statistics", "Statistics"),
        ]

        for label, page_name in menu_items:
            if st.button(
                label,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if st.session_state.page == page_name else "secondary",
            ):
                st.session_state.page = page_name
                st.rerun()

        st.markdown("---")

        # Model status
        model = get_model()
        if model:
            st.success("Model loaded")
        else:
            st.warning("Model not found - using mock inference")

        # Filters for New Analysis (in sidebar so main area is full width)
        if st.session_state.page == "New Analysis":
            st.subheader("Filters")
            _render_analysis_filters()
            if USE_MOCK_FETCHER:
                with st.expander("Using mock data"):
                    st.caption("Set FIRE_USE_MOCK=false for real Sentinel-2 imagery.")
            st.markdown("---")
            storage = get_storage()
            stats = storage.get_statistics()
            st.metric("Total Analyses", stats["total_analyses"])
            st.metric("Fire Detections", stats["fire_detections"])
        else:
            # Mock fetcher warning (only when not on New Analysis)
            if USE_MOCK_FETCHER:
                st.warning("Using mock satellite data")
                with st.expander("How to use real data"):
                    st.markdown("""
                    To fetch real Sentinel-2 imagery:

                    **Option 1: Environment variable**
                    ```bash
                    export FIRE_USE_MOCK=false
                    streamlit run app.py
                    ```

                    **Option 2: Edit app.py**
                    Set `USE_MOCK_FETCHER = False`

                    Requires `planetary-computer` package.
                    """)

            # Storage info
            storage = get_storage()
            stats = storage.get_statistics()
            st.metric("Total Analyses", stats["total_analyses"])
            st.metric("Fire Detections", stats["fire_detections"])

    # Main content
    page = st.session_state.page
    if page == "New Analysis":
        render_new_analysis()
    elif page == "History":
        render_history()
    else:
        render_statistics()


def _render_analysis_filters() -> None:
    """Render analysis filters in sidebar. Stores bbox in session state for map."""
    input_method = st.radio(
        "Input method",
        ["Coordinates", "Preset Locations"],
        horizontal=False,
    )

    if input_method == "Coordinates":
        center_lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-122.0,
            step=0.1,
        )
        center_lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=37.5,
            step=0.1,
        )
        region_size = st.slider(
            "Region Size (°)",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )
        half_size = region_size / 2
        bbox = (
            center_lon - half_size,
            center_lat - half_size,
            center_lon + half_size,
            center_lat + half_size,
        )
    else:
        # Catalonia first (default), then other regions. Bbox: (west, south, east, north)
        presets = {
            # Catalonia (Spain) - provinces spanning the region
            "Catalonia › All": (0.15, 40.5, 3.35, 42.9),
            "Catalonia › Barcelona": (1.5, 41.2, 2.5, 41.7),
            "Catalonia › Girona": (2.5, 41.7, 3.2, 42.5),
            "Catalonia › Tarragona": (0.8, 40.8, 1.8, 41.5),
            "Catalonia › Lleida": (0.3, 41.5, 1.5, 42.0),
            # Other regions
            "California Coast": (-122.5, 37.0, -121.5, 38.0),
            "Portugal": (-8.5, 38.5, -7.5, 39.5),
            "Greece": (22.5, 37.5, 23.5, 38.5),
            "Australia (NSW)": (149.5, -34.0, 150.5, -33.0),
        }
        preset_keys = list(presets.keys())
        location = st.selectbox("Location", preset_keys, index=0)
        bbox = presets[location]
        center_lon = (bbox[0] + bbox[2]) / 2
        center_lat = (bbox[1] + bbox[3]) / 2

    st.caption(f"Bbox: ({bbox[0]:.2f}, {bbox[1]:.2f}) → ({bbox[2]:.2f}, {bbox[3]:.2f})")
    st.session_state.analysis_bbox = bbox
    st.session_state.analysis_center = (center_lat, center_lon)

    # Map in sidebar
    try:
        import folium
        from streamlit_folium import st_folium

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=9,
            tiles="OpenStreetMap",
        )
        folium.Rectangle(
            bounds=[[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
            color="red",
            fill=True,
            fillOpacity=0.2,
        ).add_to(m)
        st_folium(m, height=220)
    except ImportError:
        pass

    st.markdown("---")
    st.caption("Date range")
    end_date = st.date_input(
        "End Date",
        value=datetime.now().date(),
        max_value=datetime.now().date(),
    )
    days_back = st.slider("Days to search", min_value=1, max_value=60, value=30)
    start_date = end_date - timedelta(days=days_back)
    max_cloud = st.slider("Max Cloud (%)", min_value=5, max_value=50, value=20)

    st.markdown("---")
    if st.button("🛰️ Fetch & Analyze", type="primary", use_container_width=True):
        run_analysis(
            bbox=bbox,
            date_range=(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
            max_cloud_cover=max_cloud,
        )
        st.rerun()


def render_new_analysis():
    """Render the new analysis page - full-width main area for results."""
    st.subheader("Results")

    # Results - full width main area
    if "analysis_result" in st.session_state and st.session_state.analysis_result:
        render_analysis_results(st.session_state.analysis_result)


def run_analysis(
    bbox: tuple[float, float, float, float],
    date_range: tuple[str, str],
    max_cloud_cover: float,
):
    """Fetch satellite data, run fire detection, store results in session state."""
    storage = get_storage()
    model = get_model()

    with st.spinner("Fetching satellite imagery..."):
        fetcher = get_fetcher(use_mock=USE_MOCK_FETCHER)

        try:
            image = fetcher.fetch_region(
                bbox=bbox,
                date_range=date_range,
                max_cloud_cover=max_cloud_cover,
            )
        except Exception as e:
            st.error(f"Failed to fetch imagery: {e}")
            return

        if image is None:
            st.warning("No suitable imagery found for the selected region and date range.")
            return

    # Run inference
    with st.spinner("Running fire detection..."):
        if model:
            result = model.predict_from_array(image.data, select_bands=False)
        else:
            result = run_inference_mock(image)

    # Create visualization
    if model:
        visualization = create_visualization(image.data, result)
    else:
        rgb = create_rgb_preview(image)
        visualization = rgb.copy()
        fire_mask = result.segmentation > 0
        visualization[fire_mask] = [255, 0, 0]  # Red overlay for fire

    # Save to storage
    with st.spinner("Saving analysis..."):
        analysis_id = storage.save_analysis(
            satellite_image=image,
            inference_result=result,
            visualization=visualization,
            metadata={"bbox": list(bbox), "date_range": list(date_range)},
        )

    # Store in session state for rendering in main area
    st.session_state.analysis_result = {
        "image": image,
        "result": result,
        "visualization": visualization,
        "analysis_id": analysis_id,
    }


def render_analysis_results(data: dict) -> None:
    """Render analysis results in the main content area."""
    image = data["image"]
    result = data["result"]
    visualization = data["visualization"]
    analysis_id = data["analysis_id"]

    st.markdown("---")
    st.success(f"Found image: {image.scene_id}")

    # Image info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Image Date", image.datetime.strftime("%Y-%m-%d") if image.datetime else "Unknown")
    with col2:
        st.metric("Cloud Cover", f"{image.cloud_cover:.1f}%")
    with col3:
        st.metric("Resolution", f"{image.data.shape[0]}x{image.data.shape[1]}")

    st.markdown("---")
    st.subheader("Results")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fire Detected", "🔥 Yes" if result.has_fire else "✅ No")
    with col2:
        st.metric("Confidence", f"{result.fire_confidence:.1%}")
    with col3:
        st.metric("Fire Coverage", f"{result.fire_fraction:.2%}")
    with col4:
        st.metric("Classes", result.num_classes)

    # Synced zoom/pan viewer - full width in main area
    rgb_preview = create_rgb_preview(image)
    st.caption("Scroll to zoom, drag to pan. Both images stay in sync.")
    render_synced_images(rgb_preview, visualization, "Original", "Fire detection")

    # When fire detected, show high-contrast fire mask so sparse detections are visible
    if result.has_fire:
        with st.expander("📍 Where are the fires?", expanded=True):
            fire_mask_viz = create_fire_mask_visualization(
                result.segmentation,
                num_classes=result.num_classes,
                dilate_pixels=3,
            )
            st.caption("Fire pixels (red) on dark background. Dilated 3px so sparse detections are visible.")
            st.image(resize_for_display(fire_mask_viz, max_size=1000), width="stretch")

    # Severity breakdown
    if result.severity_counts:
        st.markdown("**Severity Distribution**")
        cols = st.columns(len(result.severity_counts))
        for i, (name, count) in enumerate(result.severity_counts.items()):
            with cols[i]:
                total = sum(result.severity_counts.values())
                pct = count / total * 100 if total > 0 else 0
                st.metric(name.replace("_", " ").title(), f"{pct:.1f}%")

    st.success(f"Analysis saved! ID: {analysis_id}")


def render_history():
    """Render the history page."""
    st.header("Analysis History")

    storage = get_storage()

    # Filters
    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            fire_only = st.checkbox("Fire detections only")

        with col2:
            date_filter = st.date_input(
                "Date range",
                value=(datetime.now().date() - timedelta(days=30), datetime.now().date()),
            )

        with col3:
            limit = st.selectbox("Show", [10, 25, 50, 100], index=1)

    # Parse date filter
    if isinstance(date_filter, tuple) and len(date_filter) == 2:
        start_date = datetime.combine(date_filter[0], datetime.min.time())
        end_date = datetime.combine(date_filter[1], datetime.max.time())
    else:
        start_date = None
        end_date = None

    # Load history
    history = storage.get_history(
        limit=limit,
        fire_only=fire_only,
        start_date=start_date,
        end_date=end_date,
    )

    if not history:
        st.info("No analyses found. Run a new analysis to see history.")
        return

    # Display as cards
    for record in history:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

            with col1:
                st.markdown(f"**{record.scene_id}**")
                st.caption(record.created_at.strftime("%Y-%m-%d %H:%M") if record.created_at else "Unknown")

            with col2:
                if record.has_fire:
                    st.markdown("🔥 **Fire**")
                else:
                    st.markdown("✅ Clear")

            with col3:
                st.markdown(f"Conf: {record.fire_confidence:.0%}")

            with col4:
                st.markdown(f"☁️ {record.cloud_cover:.0f}%")

            with col5:
                if st.button("View", key=f"view_{record.id}"):
                    st.session_state["view_analysis"] = record.id

            st.markdown("---")

    # View selected analysis
    if "view_analysis" in st.session_state:
        analysis_id = st.session_state["view_analysis"]
        analysis = storage.load_analysis(analysis_id)

        if analysis:
            st.subheader(f"Analysis: {analysis_id}")

            record = analysis["record"]

            # Give images 80% width, details 20%
            img_col, details_col = st.columns([4, 1])

            with img_col:
                if "visualization" in analysis and "image" in analysis:
                    rgb = analysis["image"][:, :, [2, 1, 0]]
                    rgb = np.clip(rgb * 3.0, 0, 1)
                    rgb_uint8 = (rgb * 255).astype(np.uint8)
                    st.caption("Scroll to zoom, drag to pan.")
                    render_synced_images(
                        rgb_uint8,
                        analysis["visualization"],
                        "Original",
                        "Fire detection",
                    )
                    if record.has_fire and "segmentation" in analysis:
                        with st.expander("📍 Where are the fires?", expanded=False):
                            fire_mask_viz = create_fire_mask_visualization(
                                analysis["segmentation"],
                                dilate_pixels=3,
                            )
                            st.image(
                                resize_for_display(fire_mask_viz, max_size=1000),
                                width="stretch",
                            )
                elif "visualization" in analysis:
                    viz = resize_for_display(analysis["visualization"])
                    st.image(viz, width="stretch")
                elif "image" in analysis:
                    rgb = analysis["image"][:, :, [2, 1, 0]]
                    rgb = np.clip(rgb * 3.0, 0, 1)
                    rgb_uint8 = (rgb * 255).astype(np.uint8)
                    st.image(resize_for_display(rgb_uint8), width="stretch")

            with details_col:
                st.markdown(f"**Scene:** {record.scene_id}")
                st.markdown(f"**Date:** {record.satellite_datetime}")
                st.markdown(f"**Location:** ({record.center_lon:.2f}, {record.center_lat:.2f})")
                st.markdown(f"**Fire Detected:** {'Yes' if record.has_fire else 'No'}")
                st.markdown(f"**Confidence:** {record.fire_confidence:.1%}")
                st.markdown(f"**Coverage:** {record.fire_fraction:.2%}")

                if st.button("Delete Analysis"):
                    storage.delete_analysis(analysis_id)
                    del st.session_state["view_analysis"]
                    st.rerun()

                if st.button("Close"):
                    del st.session_state["view_analysis"]
                    st.rerun()


def render_statistics():
    """Render the statistics page."""
    st.header("Statistics")

    storage = get_storage()
    stats = storage.get_statistics()

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Analyses", stats["total_analyses"])

    with col2:
        st.metric("Fire Detections", stats["fire_detections"])

    with col3:
        detection_rate = (
            stats["fire_detections"] / stats["total_analyses"] * 100
            if stats["total_analyses"] > 0
            else 0
        )
        st.metric("Detection Rate", f"{detection_rate:.1f}%")

    with col4:
        st.metric("Avg Cloud Cover", f"{stats['avg_cloud_cover']:.1f}%" if stats['avg_cloud_cover'] else "N/A")

    # Time range
    if stats["first_analysis"] and stats["last_analysis"]:
        st.markdown("---")
        st.markdown(f"**Analysis Period:** {stats['first_analysis'][:10]} to {stats['last_analysis'][:10]}")

    # Recent detections
    st.markdown("---")
    st.subheader("Recent Fire Detections")

    recent_fires = storage.get_history(limit=10, fire_only=True)

    if recent_fires:
        for record in recent_fires:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.markdown(f"**{record.scene_id}**")
                st.caption(f"({record.center_lon:.2f}, {record.center_lat:.2f})")

            with col2:
                st.markdown(f"{record.fire_confidence:.0%}")

            with col3:
                if record.created_at:
                    st.caption(record.created_at.strftime("%m/%d"))
    else:
        st.info("No fire detections recorded yet.")

    # Maintenance
    st.markdown("---")
    st.subheader("Maintenance")

    col1, col2 = st.columns(2)

    with col1:
        days = st.number_input("Delete analyses older than (days)", min_value=7, value=90)

    with col2:
        if st.button("🗑️ Clean Old Data"):
            deleted = storage.clear_old_analyses(days=days)
            st.success(f"Deleted {deleted} old analyses")
            st.rerun()


if __name__ == "__main__":
    main()
