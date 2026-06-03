"""Interactive on-screen control panel for live visualization tuning.

A lightweight immediate-style UI built only on pyglet primitives (shapes +
labels). It exposes sliders, toggles, and cyclers bound to getter/setter pairs
so the replay app can wire them to :class:`~boidforge.viz.renderer.RenderConfig`
and the camera without this module knowing about either. It draws on top of the
ModernGL frame; it imports neither ModernGL nor the solver.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pyglet


class ControlKind(enum.Enum):
    """The interaction style of a :class:`Control`.

    Attributes:
        SLIDER: Continuous float dragged along a track.
        INT: Integer slider (snaps to whole values).
        TOGGLE: Boolean on/off box.
        CHOICE: Cycle through a fixed list of string options.
    """

    SLIDER = "slider"
    INT = "int"
    TOGGLE = "toggle"
    CHOICE = "choice"


@dataclass(slots=True)
class Control:
    """A single tunable bound to a getter/setter.

    Attributes:
        label: Display name.
        kind: Interaction style (see :class:`ControlKind`).
        get: Returns the current value.
        set: Applies a new value.
        lo: Minimum (sliders only).
        hi: Maximum (sliders only).
        fmt: ``format`` spec for the numeric readout.
        options: Allowed strings (CHOICE only).
    """

    label: str
    kind: ControlKind
    get: Callable[[], object]
    set: Callable[[object], None]
    lo: float = 0.0
    hi: float = 1.0
    fmt: str = "{:.2f}"
    options: Sequence[str] = field(default_factory=tuple)


# Layout constants (pixels). Origin is the pyglet bottom-left convention.
_PANEL_W = 264
_PAD = 14
_ROW_H = 40
_TRACK_H = 5
_KNOB_R = 7
_TITLE_H = 30


class ControlPanel:
    """A draggable-slider control panel anchored to the top-right corner."""

    def __init__(self, controls: Sequence[Control], window_w: int, window_h: int) -> None:
        """Build the panel's persistent shapes and labels.

        Args:
            controls: The tunables to expose, top to bottom.
            window_w: Current window width in pixels.
            window_h: Current window height in pixels.
        """
        self._controls = list(controls)
        self._win_w = window_w
        self._win_h = window_h
        self.visible = True
        self._active: int | None = None  # index of slider being dragged

        self._batch = pyglet.graphics.Batch()
        self._bg_group = pyglet.graphics.Group(order=0)
        self._fg_group = pyglet.graphics.Group(order=1)
        self._txt_group = pyglet.graphics.Group(order=2)

        n = len(self._controls)
        panel_h = _TITLE_H + n * _ROW_H + _PAD
        self._panel_h = panel_h

        self._bg = pyglet.shapes.Rectangle(
            0, 0, _PANEL_W, panel_h, color=(12, 14, 22), batch=self._batch, group=self._bg_group
        )
        self._bg.opacity = 205
        self._title = pyglet.text.Label(
            "VISUAL CONTROLS",
            font_name="Consolas",
            font_size=10,
            weight="bold",
            color=(150, 200, 255, 255),
            batch=self._batch,
            group=self._txt_group,
        )

        # One track, knob, name label and value label per control.
        self._tracks: list[pyglet.shapes.Rectangle] = []
        self._fills: list[pyglet.shapes.Rectangle] = []
        self._knobs: list[pyglet.shapes.Circle] = []
        self._names: list[pyglet.text.Label] = []
        self._values: list[pyglet.text.Label] = []
        for _ in self._controls:
            self._tracks.append(
                pyglet.shapes.Rectangle(
                    0,
                    0,
                    10,
                    _TRACK_H,
                    color=(60, 66, 84),
                    batch=self._batch,
                    group=self._fg_group,
                )
            )
            self._fills.append(
                pyglet.shapes.Rectangle(
                    0,
                    0,
                    10,
                    _TRACK_H,
                    color=(90, 170, 255),
                    batch=self._batch,
                    group=self._fg_group,
                )
            )
            self._knobs.append(
                pyglet.shapes.Circle(
                    0,
                    0,
                    _KNOB_R,
                    color=(180, 220, 255),
                    batch=self._batch,
                    group=self._txt_group,
                )
            )
            self._names.append(
                pyglet.text.Label(
                    "",
                    font_name="Consolas",
                    font_size=9,
                    color=(210, 215, 230, 255),
                    batch=self._batch,
                    group=self._txt_group,
                )
            )
            self._values.append(
                pyglet.text.Label(
                    "",
                    font_name="Consolas",
                    font_size=9,
                    color=(120, 200, 160, 255),
                    anchor_x="right",
                    batch=self._batch,
                    group=self._txt_group,
                )
            )
        self._layout()
        self.refresh()

    # -- geometry ----------------------------------------------------------

    def _origin(self) -> tuple[int, int]:
        """Bottom-left corner of the panel in window coordinates."""
        x = self._win_w - _PANEL_W - _PAD
        y = self._win_h - self._panel_h - _PAD
        return x, y

    def _row_track_rect(self, i: int) -> tuple[float, float, float]:
        """Return ``(track_x, track_y, track_w)`` for control row ``i``."""
        ox, oy = self._origin()
        track_x = ox + _PAD
        track_w = _PANEL_W - 2 * _PAD
        top = oy + self._panel_h - _TITLE_H
        row_top = top - i * _ROW_H
        track_y = row_top - _ROW_H + 18
        return track_x, track_y, track_w

    def _layout(self) -> None:
        """Reposition every shape/label for the current window size."""
        ox, oy = self._origin()
        self._bg.x = ox
        self._bg.y = oy
        self._bg.height = self._panel_h
        self._title.x = ox + _PAD
        self._title.y = oy + self._panel_h - 20
        for i in range(len(self._controls)):
            tx, ty, tw = self._row_track_rect(i)
            self._tracks[i].x = tx
            self._tracks[i].y = ty
            self._tracks[i].width = tw
            self._fills[i].x = tx
            self._fills[i].y = ty
            self._knobs[i].y = ty + _TRACK_H / 2
            self._names[i].x = tx
            self._names[i].y = ty + 12
            self._values[i].x = tx + tw
            self._values[i].y = ty + 12

    def resize(self, window_w: int, window_h: int) -> None:
        """Re-anchor the panel after a window resize.

        Args:
            window_w: New window width in pixels.
            window_h: New window height in pixels.
        """
        self._win_w = window_w
        self._win_h = window_h
        self._layout()
        self.refresh()

    # -- value sync --------------------------------------------------------

    def _normalized(self, c: Control) -> float:
        """Current value of ``c`` mapped to ``[0, 1]`` for the track."""
        if c.kind in (ControlKind.SLIDER, ControlKind.INT):
            span = c.hi - c.lo or 1.0
            return min(1.0, max(0.0, (float(c.get()) - c.lo) / span))  # type: ignore[arg-type]
        if c.kind is ControlKind.TOGGLE:
            return 1.0 if c.get() else 0.0
        # CHOICE: position within the option list.
        opts = list(c.options)
        cur = str(c.get())
        idx = opts.index(cur) if cur in opts else 0
        return idx / max(len(opts) - 1, 1)

    def _format_value(self, c: Control) -> str:
        """Human-readable readout for ``c``'s current value."""
        if c.kind is ControlKind.TOGGLE:
            return "ON" if c.get() else "OFF"
        if c.kind is ControlKind.CHOICE:
            return str(c.get())
        return c.fmt.format(c.get())

    def refresh(self) -> None:
        """Pull current values from getters and update labels/knobs."""
        for i, c in enumerate(self._controls):
            tx, ty, tw = self._row_track_rect(i)
            t = self._normalized(c)
            self._fills[i].width = max(1.0, tw * t)
            self._knobs[i].x = tx + tw * t
            self._names[i].text = c.label
            self._values[i].text = self._format_value(c)

    # -- interaction -------------------------------------------------------

    def _hit_row(self, x: float, y: float) -> int | None:
        """Index of the control row under ``(x, y)``, or ``None``."""
        for i in range(len(self._controls)):
            tx, ty, tw = self._row_track_rect(i)
            if tx - _KNOB_R <= x <= tx + tw + _KNOB_R and ty - 14 <= y <= ty + 20:
                return i
        return None

    def contains(self, x: float, y: float) -> bool:
        """Whether the point lies within the (visible) panel rectangle.

        Args:
            x: Cursor X in window coordinates.
            y: Cursor Y in window coordinates.

        Returns:
            True if the panel is visible and the point is inside it.
        """
        if not self.visible:
            return False
        ox, oy = self._origin()
        return ox <= x <= ox + _PANEL_W and oy <= y <= oy + self._panel_h

    def on_press(self, x: float, y: float) -> bool:
        """Handle a mouse press; returns True if the panel consumed it.

        Args:
            x: Cursor X in window coordinates.
            y: Cursor Y in window coordinates.

        Returns:
            True if a control was activated (so the app should not pan/zoom).
        """
        if not self.visible:
            return False
        i = self._hit_row(x, y)
        if i is None:
            return self.contains(x, y)
        c = self._controls[i]
        if c.kind is ControlKind.TOGGLE:
            c.set(not c.get())
        elif c.kind is ControlKind.CHOICE:
            opts = list(c.options)
            cur = str(c.get())
            nxt = opts[(opts.index(cur) + 1) % len(opts)] if cur in opts else opts[0]
            c.set(nxt)
        else:
            self._active = i
            self._drag_to(i, x)
        self.refresh()
        return True

    def on_drag(self, x: float, y: float) -> bool:
        """Handle a mouse drag; returns True if dragging a slider.

        Args:
            x: Cursor X in window coordinates.
            y: Cursor Y in window coordinates (unused; horizontal tracks).

        Returns:
            True if a slider is being dragged.
        """
        if self._active is None:
            return False
        self._drag_to(self._active, x)
        self.refresh()
        return True

    def on_release(self) -> None:
        """Release any slider currently being dragged."""
        self._active = None

    def _drag_to(self, i: int, x: float) -> None:
        """Set control ``i`` from a cursor X along its track."""
        c = self._controls[i]
        tx, _, tw = self._row_track_rect(i)
        t = min(1.0, max(0.0, (x - tx) / (tw or 1.0)))
        value = c.lo + t * (c.hi - c.lo)
        if c.kind is ControlKind.INT:
            c.set(int(round(value)))
        else:
            c.set(float(value))

    # -- draw --------------------------------------------------------------

    def draw(self) -> None:
        """Draw the panel if visible."""
        if self.visible:
            self._batch.draw()


class InfoHUD:
    """Top-left status readout: frame, playback state, fps, and key help."""

    def __init__(self, window_h: int) -> None:
        """Create the HUD labels.

        Args:
            window_h: Current window height in pixels (for top anchoring).
        """
        self._win_h = window_h
        self._batch = pyglet.graphics.Batch()
        self._status = pyglet.text.Label(
            "",
            font_name="Consolas",
            font_size=11,
            weight="bold",
            color=(230, 240, 255, 255),
            x=14,
            y=window_h - 22,
            batch=self._batch,
        )
        self._help = pyglet.text.Label(
            "SPACE play/pause  ←→ step  ↑↓ speed  R restart  "
            "F follow  H panel  C colormap  drag pan  wheel zoom  ESC quit",
            font_name="Consolas",
            font_size=9,
            color=(150, 160, 180, 220),
            x=14,
            y=14,
            batch=self._batch,
        )
        self.visible = True

    def resize(self, window_h: int) -> None:
        """Re-anchor the status line after a window resize.

        Args:
            window_h: New window height in pixels.
        """
        self._win_h = window_h
        self._status.y = window_h - 22

    def set_status(self, text: str) -> None:
        """Update the status line text.

        Args:
            text: The new status string.
        """
        self._status.text = text

    def draw(self) -> None:
        """Draw the HUD if visible."""
        if self.visible:
            self._batch.draw()
