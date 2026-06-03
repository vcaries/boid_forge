"""Interactive pyglet window: live replay with on-screen parameter tuning.

Ties together the ModernGL :class:`~boidforge.viz.renderer.Renderer`, the
:class:`~boidforge.viz.camera.Camera`, and the
:class:`~boidforge.viz.ui.ControlPanel` into a windowed player. Playback
(play/pause, step, speed, scrub) and camera (pan/zoom/follow) are driven by
mouse and keyboard; every visual parameter is adjustable live through the panel.

This module imports pyglet and ModernGL but never the solver. It advances no
physics — it only replays frames already decoded from a ``.bfs`` stream.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pyglet
from pyglet import gl
from pyglet.window import key

from boidforge.io.reader import Frame
from boidforge.viz import colormaps
from boidforge.viz.camera import Camera
from boidforge.viz.renderer import ColorMode, RenderConfig, Renderer
from boidforge.viz.ui import Control, ControlKind, ControlPanel, InfoHUD

_COLOR_MODES = ("speed", "heading", "uniform")


class InteractiveApp:
    """A windowed, interactively tunable replay of a decoded frame sequence."""

    def __init__(
        self,
        frames: Sequence[Frame],
        config: RenderConfig,
        world: tuple[float, float],
        fps: int = 60,
        loop: bool = True,
    ) -> None:
        """Open a window, bind the GL context, and wire up controls.

        Args:
            frames: Decoded frames to replay (already in memory).
            config: Initial appearance settings (mutated live by the panel).
            world: ``(world_width, world_height)`` for the initial camera fit.
            fps: Target playback frame rate.
            loop: Whether to wrap to the start at the end of the stream.
        """
        if not frames:
            raise ValueError("no frames to replay")
        self._frames = list(frames)
        self.config = config
        self._fps = fps
        self._loop = loop

        self._pos = 0.0  # fractional frame cursor
        self._speed = 1.0  # frames advanced per tick (signed)
        self._playing = True

        self._window = pyglet.window.Window(
            width=config.width,
            height=config.height,
            caption="BoidForge — Replay",
            resizable=True,
            vsync=True,
        )
        self._window.switch_to()
        import moderngl

        self._ctx = moderngl.create_context()
        fb_w, fb_h = self._window.get_framebuffer_size()
        config.width, config.height = fb_w, fb_h
        self._renderer = Renderer(config, ctx=self._ctx, max_boids=self._max_n())
        self._camera = Camera()
        self._camera.frame_world(world[0], world[1], fb_w, fb_h)

        self._panel = ControlPanel(self._build_controls(), self._window.width, self._window.height)
        self._hud = InfoHUD(self._window.height)
        self._dragging_cam = False

        self._window.push_handlers(
            on_draw=self._on_draw,
            on_resize=self._on_resize,
            on_mouse_press=self._on_mouse_press,
            on_mouse_release=self._on_mouse_release,
            on_mouse_drag=self._on_mouse_drag,
            on_mouse_scroll=self._on_mouse_scroll,
            on_key_press=self._on_key_press,
        )
        pyglet.clock.schedule_interval(self._tick, 1.0 / fps)

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        """Enter the pyglet event loop until the window is closed."""
        try:
            pyglet.app.run()
        finally:
            self._renderer.release()

    def _max_n(self) -> int:
        """Largest boid count across all frames (vertex-buffer sizing)."""
        return max(f.state.n for f in self._frames)

    # -- playback ----------------------------------------------------------

    def _tick(self, dt: float) -> None:
        """Advance the frame cursor (called on the playback clock)."""
        del dt
        if not self._playing:
            return
        self._pos += self._speed
        last = len(self._frames) - 1
        if self._pos > last:
            self._pos = 0.0 if self._loop else float(last)
            if not self._loop:
                self._playing = False
        elif self._pos < 0:
            self._pos = float(last) if self._loop else 0.0

    def _current_index(self) -> int:
        """Clamped integer index of the frame to display."""
        return max(0, min(len(self._frames) - 1, int(self._pos)))

    # -- rendering ---------------------------------------------------------

    def _on_draw(self) -> None:
        """Render the current frame, then overlay the UI."""
        frame = self._frames[self._current_index()]
        self._camera.update(frame.state)
        self._renderer.draw(frame.state, self._camera, present=True)

        # Restore the GL state pyglet's batched UI assumes (ModernGL changed it).
        self._ctx.screen.use()
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST)
        self._panel.refresh()
        self._hud.set_status(self._status_text())
        self._panel.draw()
        self._hud.draw()

    def _status_text(self) -> str:
        """One-line status: frame counter, play state, and speed."""
        idx = self._current_index()
        state = "▶" if self._playing else "❚❚"
        follow = "follow" if self._camera.follow else "free"
        return (
            f"{state}  frame {idx + 1}/{len(self._frames)}  "
            f"speed x{self._speed:+.2f}  zoom {self._camera.zoom:.2f}  {follow}  "
            f"N={self._frames[idx].state.n}"
        )

    # -- events ------------------------------------------------------------

    def _on_resize(self, width: int, height: int) -> None:
        """Resize render targets and re-anchor the UI."""
        fb_w, fb_h = self._window.get_framebuffer_size()
        self._renderer.resize(fb_w, fb_h)
        self._panel.resize(self._window.width, self._window.height)
        self._hud.resize(self._window.height)

    def _on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        """Route a click to the panel, else begin a camera pan."""
        del button, modifiers
        if self._panel.on_press(x, y):
            return
        self._dragging_cam = True

    def _on_mouse_release(self, x: int, y: int, button: int, modifiers: int) -> None:
        """End slider/camera dragging."""
        del x, y, button, modifiers
        self._panel.on_release()
        self._dragging_cam = False

    def _on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int
    ) -> None:
        """Drag a slider, or pan the camera (which disables follow)."""
        del buttons, modifiers
        if self._panel.on_drag(x, y):
            return
        if self._dragging_cam:
            self._camera.follow = False
            self._camera.pan_pixels(dx, dy)

    def _on_mouse_scroll(self, x: int, y: int, sx: float, sy: float) -> None:
        """Zoom the camera around the cursor."""
        del x, y, sx
        self._camera.zoom_by(1.0 + 0.12 * sy)

    def _on_key_press(self, symbol: int, modifiers: int) -> None:
        """Keyboard controls for playback, camera, and panel visibility."""
        del modifiers
        if symbol == key.SPACE:
            self._playing = not self._playing
        elif symbol == key.RIGHT:
            self._playing = False
            self._pos = min(self._pos + 1, len(self._frames) - 1)
        elif symbol == key.LEFT:
            self._playing = False
            self._pos = max(self._pos - 1, 0)
        elif symbol == key.UP:
            self._speed = min(self._speed * 1.25 if self._speed > 0 else 0.25, 8.0)
        elif symbol == key.DOWN:
            self._speed = max(self._speed / 1.25, 0.05)
        elif symbol == key.R:
            self._pos = 0.0
            self._playing = True
            self._renderer.clear_trails()
        elif symbol == key.F:
            self._camera.follow = not self._camera.follow
        elif symbol == key.H:
            self._panel.visible = not self._panel.visible
        elif symbol == key.C:
            self._cycle_colormap()
        elif symbol == key.ESCAPE:
            self._window.close()

    def _cycle_colormap(self) -> None:
        """Advance to the next available colormap."""
        names = list(colormaps.available())
        cur = self.config.colormap
        nxt = names[(names.index(cur) + 1) % len(names)] if cur in names else names[0]
        self._renderer.set_colormap(nxt)

    # -- control bindings --------------------------------------------------

    def _build_controls(self) -> list[Control]:
        """Build the panel's control list bound to config/camera/playback."""
        cfg = self.config

        def fattr(attr: str, lo: float, hi: float, fmt: str = "{:.2f}") -> Control:
            # ``attr`` is fixed per call, so a plain closure captures it safely.
            def get() -> object:
                return getattr(cfg, attr)

            def set_(v: object) -> None:
                setattr(cfg, attr, float(cast(float, v)))

            return Control(
                label=attr.replace("_", " ").title(),
                kind=ControlKind.SLIDER,
                get=get,
                set=set_,
                lo=lo,
                hi=hi,
                fmt=fmt,
            )

        def toggle(label: str, attr: str) -> Control:
            def get() -> object:
                return getattr(cfg, attr)

            def set_(v: object) -> None:
                setattr(cfg, attr, bool(v))

            return Control(label=label, kind=ControlKind.TOGGLE, get=get, set=set_)

        controls: list[Control] = [
            fattr("point_size", 1.0, 40.0, "{:.1f}"),
            fattr("glow", 0.0, 2.0),
            fattr("intensity", 0.2, 3.0),
            fattr("trail_decay", 0.0, 0.99),
            toggle("Bloom", "bloom"),
            fattr("bloom_strength", 0.0, 3.0),
            fattr("bloom_threshold", 0.0, 2.0),
            fattr("exposure", 0.2, 3.0),
            fattr("vignette", 0.0, 1.0),
            fattr("speed_ref", 20.0, 400.0, "{:.0f}"),
            Control(
                label="Color By",
                kind=ControlKind.CHOICE,
                get=lambda: ColorMode(self.config.color_mode).name.lower(),
                set=self._set_color_mode,
                options=_COLOR_MODES,
            ),
            Control(
                label="Colormap",
                kind=ControlKind.CHOICE,
                get=lambda: self.config.colormap,
                set=lambda v: self._renderer.set_colormap(str(v)),
                options=colormaps.available(),
            ),
            Control(
                label="Zoom",
                kind=ControlKind.SLIDER,
                get=lambda: self._camera.zoom,
                set=lambda v: setattr(self._camera, "zoom", float(cast(float, v))),
                lo=0.05,
                hi=4.0,
            ),
            Control(
                label="Follow Flock",
                kind=ControlKind.TOGGLE,
                get=lambda: self._camera.follow,
                set=lambda v: setattr(self._camera, "follow", bool(v)),
            ),
            Control(
                label="Play Speed",
                kind=ControlKind.SLIDER,
                get=lambda: self._speed,
                set=lambda v: setattr(self, "_speed", float(cast(float, v))),
                lo=0.05,
                hi=8.0,
            ),
        ]
        return controls

    def _set_color_mode(self, value: object) -> None:
        """Set the colour mode from its lowercase name."""
        self.config.color_mode = ColorMode[str(value).upper()]
