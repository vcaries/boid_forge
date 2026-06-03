"""Headless rendering and visualization-math tests.

The GPU passes are exercised through an offscreen ModernGL standalone context.
If no GL driver is available (e.g. a headless CI box without EGL), the GPU tests
skip cleanly; the pure-math tests (camera, colormaps) always run.
"""

from __future__ import annotations

import numpy as np
import pytest

from boidforge.core.state import SimulationState
from boidforge.core.types import DTYPE
from boidforge.viz import colormaps
from boidforge.viz.camera import Camera


def _make_state(n: int, seed: int = 0) -> SimulationState:
    """Build a small random SoA state for rendering tests."""
    rng = np.random.default_rng(seed)
    return SimulationState(
        px=(rng.random(n) * 1920).astype(DTYPE),
        py=(rng.random(n) * 1080).astype(DTYPE),
        vx=((rng.random(n) - 0.5) * 200).astype(DTYPE),
        vy=((rng.random(n) - 0.5) * 200).astype(DTYPE),
    )


# -- colormaps -----------------------------------------------------------------


def test_colormaps_shape_and_range() -> None:
    """Every colormap is a 256x3 float32 LUT within [0, 1]."""
    for name in colormaps.available():
        lut = colormaps.lookup_table(name)
        assert lut.shape == (colormaps.LUT_SIZE, 3)
        assert lut.dtype == np.float32
        assert float(lut.min()) >= 0.0 and float(lut.max()) <= 1.0


def test_colormap_unknown_raises() -> None:
    """An unknown colormap name is rejected."""
    with pytest.raises(KeyError):
        colormaps.lookup_table("does-not-exist")


# -- camera --------------------------------------------------------------------


def test_camera_follow_moves_toward_com() -> None:
    """Following the flock moves the centre toward the mean position."""
    state = _make_state(256, seed=1)
    com = (float(np.mean(state.px)), float(np.mean(state.py)))
    cam = Camera(center_x=0.0, center_y=0.0, follow=True, smoothing=0.0)
    cam.update(state)  # smoothing 0 => instant snap
    assert cam.center_x == pytest.approx(com[0], rel=1e-5)
    assert cam.center_y == pytest.approx(com[1], rel=1e-5)


def test_camera_no_follow_is_static() -> None:
    """With follow disabled the centre is untouched by update()."""
    state = _make_state(64)
    cam = Camera(center_x=10.0, center_y=20.0, follow=False)
    cam.update(state)
    assert (cam.center_x, cam.center_y) == (10.0, 20.0)


def test_view_matrix_maps_center_to_origin() -> None:
    """The view centre maps to the clip-space origin."""
    cam = Camera(center_x=960.0, center_y=540.0, zoom=1.0, follow=False)
    m = cam.view_matrix(1920, 1080)
    # Column-major 4x4 applied to (cx, cy, 0, 1).
    cx, cy = 960.0, 540.0
    clip_x = m[0] * cx + m[4] * cy + m[12]
    clip_y = m[1] * cx + m[5] * cy + m[13]
    assert clip_x == pytest.approx(0.0, abs=1e-4)
    assert clip_y == pytest.approx(0.0, abs=1e-4)


def test_zoom_by_clamps_positive() -> None:
    """Zoom stays positive and ignores non-positive factors."""
    cam = Camera(zoom=1.0)
    cam.zoom_by(2.0)
    assert cam.zoom == pytest.approx(2.0)
    cam.zoom_by(-1.0)  # ignored
    assert cam.zoom == pytest.approx(2.0)


# -- offscreen rendering -------------------------------------------------------


@pytest.fixture(scope="module")
def gl_available() -> bool:
    """Whether a standalone ModernGL context can be created here."""
    try:
        import moderngl

        ctx = moderngl.create_standalone_context()
        ctx.release()
        return True
    except Exception:
        return False


def test_offscreen_render_produces_pixels(gl_available: bool) -> None:
    """A few rendered frames yield a correctly sized, non-empty image."""
    if not gl_available:
        pytest.skip("no GL driver for offscreen rendering")
    from boidforge.viz.renderer import RenderConfig, Renderer

    cfg = RenderConfig(width=320, height=180)
    renderer = Renderer(cfg, ctx=None, max_boids=512)
    cam = Camera()
    cam.frame_world(1920, 1080, 320, 180)
    try:
        for i in range(8):
            renderer.draw(_make_state(400, seed=i), cam, present=False)
        pixels = renderer.read_pixels()
        assert len(pixels) == 320 * 180 * 4
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(180, 320, 4)
        assert int(arr[..., :3].max()) > 16  # something was drawn
    finally:
        renderer.release()


def test_renderer_grows_buffer(gl_available: bool) -> None:
    """Rendering more boids than the initial capacity grows the buffer."""
    if not gl_available:
        pytest.skip("no GL driver for offscreen rendering")
    from boidforge.viz.renderer import RenderConfig, Renderer

    renderer = Renderer(RenderConfig(width=128, height=128), ctx=None, max_boids=64)
    cam = Camera()
    cam.frame_world(1920, 1080, 128, 128)
    try:
        renderer.draw(_make_state(1000, seed=3), cam, present=False)  # forces grow
        assert len(renderer.read_pixels()) == 128 * 128 * 4
    finally:
        renderer.release()
