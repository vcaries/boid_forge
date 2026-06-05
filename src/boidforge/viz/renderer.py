"""ModernGL renderer: glowing point sprites with velocity colour and trails.

The renderer uploads per-frame SoA buffers to a GPU vertex buffer and draws all
boids in a single additive pass into an HDR accumulation buffer. That buffer is
faded a little each frame to produce motion trails; an optional bloom pass adds
glow, and a final ACES tonemap composites the scene onto a dark background.

ModernGL is imported lazily (inside the methods that touch the GPU) so the
package — and :class:`RenderConfig` — import fine on a headless node without a
GL driver. This module contains no simulation logic and never imports the
solver.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from boidforge.core.state import SimulationState
from boidforge.core.types import DTYPE, FloatArray
from boidforge.viz import colormaps, shaders
from boidforge.viz.camera import Camera

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime GL dependency.
    import moderngl


class ColorMode(enum.IntEnum):
    """How a boid's colour-map coordinate is derived.

    Attributes:
        SPEED: Map speed across the observed ``[speed_lo, speed_hi]`` range.
        HEADING: Map travel direction (angle) to colour.
        UNIFORM: All boids share a single colour-map coordinate.
        DENSITY: Map local crowding (neighbours per grid cell) to colour.
    """

    SPEED = 0
    HEADING = 1
    UNIFORM = 2
    DENSITY = 3


@dataclass(slots=True)
class RenderConfig:
    """Renderer appearance and quality settings (mutable for live tuning).

    Attributes:
        width: Framebuffer width in pixels.
        height: Framebuffer height in pixels.
        point_size: Base sprite diameter in pixels.
        min_point_size: Lower clamp so zoomed-out boids stay visible.
        glow: Halo strength of each sprite in ``[0, ~2]``.
        intensity: Overall emission multiplier.
        trail_decay: Per-frame trail retention in ``[0, 1]``; 0 disables trails.
        bloom: Whether to apply the bloom/glow post pass.
        bloom_strength: How much bloom is added back in composite.
        bloom_threshold: Luminance above which pixels feed the bloom.
        exposure: HDR exposure before tonemapping.
        vignette: Corner darkening in ``[0, 1]``; 0 disables.
        colormap: Name of the colour map (see :mod:`boidforge.viz.colormaps`).
        color_mode: What drives boid colour (see :class:`ColorMode`).
        uniform_t: Colour-map coordinate used when ``color_mode`` is UNIFORM.
        speed_lo: Speed (world units/s) mapped to the cold end of the ramp.
        speed_hi: Speed (world units/s) mapped to the hot end of the ramp.
        density_cell: Grid cell side (world units) for the DENSITY colour mode;
            local crowding is the boid count sharing a cell of this size.
        background: RGB base colour the trails settle onto.
    """

    width: int = 1600
    height: int = 900
    point_size: float = 9.0
    min_point_size: float = 1.5
    glow: float = 0.9
    intensity: float = 1.15
    trail_decay: float = 0.90
    bloom: bool = True
    bloom_strength: float = 1.1
    bloom_threshold: float = 0.55
    exposure: float = 1.1
    vignette: float = 0.35
    colormap: str = "turbo"
    color_mode: ColorMode = ColorMode.SPEED
    uniform_t: float = 0.6
    speed_lo: float = 40.0
    speed_hi: float = 180.0
    density_cell: float = 36.0
    background: tuple[float, float, float] = field(default=(0.015, 0.02, 0.045))


class Renderer:
    """Stateful ModernGL renderer for a single replay session."""

    def __init__(
        self,
        config: RenderConfig,
        ctx: moderngl.Context | None = None,
        max_boids: int = 4096,
    ) -> None:
        """Create the GL context, shader programs, and framebuffers.

        Args:
            config: Appearance and quality settings.
            ctx: An existing ModernGL context (e.g. from a window). If ``None``,
                a standalone offscreen context is created — used for headless
                video export and tests.
            max_boids: Initial vertex-buffer capacity; grown automatically.
        """
        import moderngl as mgl

        self._mgl = mgl
        self.config = config
        self._ctx: moderngl.Context = ctx if ctx is not None else mgl.create_standalone_context()
        self._owns_ctx = ctx is None
        self._width = config.width
        self._height = config.height

        self._point_prog = self._ctx.program(
            vertex_shader=shaders.POINT_VERTEX, fragment_shader=shaders.POINT_FRAGMENT
        )
        self._fade_prog = self._make_fs_program(shaders.SOLID_FRAGMENT)
        self._bright_prog = self._make_fs_program(shaders.BRIGHT_FRAGMENT)
        self._blur_prog = self._make_fs_program(shaders.BLUR_FRAGMENT)
        self._composite_prog = self._make_fs_program(shaders.COMPOSITE_FRAGMENT)

        # Fullscreen triangle (covers clip space) shared by every post pass.
        tri = np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype="f4")
        self._fs_vbo = self._ctx.buffer(tri.tobytes())
        self._fade_vao = self._fs_vao(self._fade_prog)
        self._bright_vao = self._fs_vao(self._bright_prog)
        self._blur_vao = self._fs_vao(self._blur_prog)
        self._composite_vao = self._fs_vao(self._composite_prog)

        # Dynamic boid vertex buffer (pos.xy, vel.xy, aux interleaved) + scratch.
        # ``aux`` carries the precomputed normalized density for DENSITY mode.
        self._capacity = max(max_boids, 1)
        self._scratch = np.zeros((self._capacity, 5), dtype=DTYPE)
        self._boid_vbo = self._ctx.buffer(reserve=self._capacity * 20, dynamic=True)
        self._boid_vao = self._ctx.vertex_array(
            self._point_prog,
            [(self._boid_vbo, "2f 2f 1f", "in_pos", "in_vel", "in_aux")],
        )

        self._lut_tex: moderngl.Texture = self._build_lut(config.colormap)
        self._lut_name = config.colormap

        # Size-dependent framebuffers built by _allocate_targets().
        self._accum_tex: moderngl.Texture
        self._accum_fbo: moderngl.Framebuffer
        self._bright_tex: moderngl.Texture
        self._bright_fbo: moderngl.Framebuffer
        self._blur_tex: list[moderngl.Texture]
        self._blur_fbo: list[moderngl.Framebuffer]
        self._out_tex: moderngl.Texture
        self._out_fbo: moderngl.Framebuffer
        self._allocate_targets(self._width, self._height)

        self._ctx.enable(mgl.PROGRAM_POINT_SIZE)

    # -- setup helpers -----------------------------------------------------

    def _make_fs_program(self, fragment: str) -> moderngl.Program:
        """Compile a fullscreen-pass program with the shared vertex shader."""
        return self._ctx.program(vertex_shader=shaders.FULLSCREEN_VERTEX, fragment_shader=fragment)

    def _fs_vao(self, prog: moderngl.Program) -> moderngl.VertexArray:
        """Bind the fullscreen-triangle buffer to ``prog``'s ``in_pos``."""
        return self._ctx.vertex_array(prog, [(self._fs_vbo, "2f", "in_pos")])

    def _build_lut(self, name: str) -> moderngl.Texture:
        """Upload a colormap as a 256x1 linear-filtered RGB texture."""
        table = colormaps.lookup_table(name)
        tex = self._ctx.texture((colormaps.LUT_SIZE, 1), 3, table.tobytes(), dtype="f4")
        tex.filter = (self._mgl.LINEAR, self._mgl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def _allocate_targets(self, width: int, height: int) -> None:
        """(Re)allocate all size-dependent framebuffers for ``width x height``."""
        ctx = self._ctx
        self._accum_tex = ctx.texture((width, height), 4, dtype="f2")
        self._accum_fbo = ctx.framebuffer(color_attachments=[self._accum_tex])

        bw, bh = max(width // 2, 1), max(height // 2, 1)
        self._bright_tex = ctx.texture((bw, bh), 4, dtype="f2")
        self._bright_fbo = ctx.framebuffer(color_attachments=[self._bright_tex])
        self._blur_tex = [ctx.texture((bw, bh), 4, dtype="f2") for _ in range(2)]
        self._blur_fbo = [ctx.framebuffer(color_attachments=[t]) for t in self._blur_tex]
        for t in (self._bright_tex, *self._blur_tex):
            t.filter = (self._mgl.LINEAR, self._mgl.LINEAR)
            t.repeat_x = False
            t.repeat_y = False

        self._out_tex = ctx.texture((width, height), 4, dtype="f1")
        self._out_fbo = ctx.framebuffer(color_attachments=[self._out_tex])

        self._accum_fbo.clear(0.0, 0.0, 0.0, 0.0)

    # -- public API --------------------------------------------------------

    @property
    def ctx(self) -> moderngl.Context:
        """The ModernGL context this renderer draws with.

        Returns:
            The active context (standalone or window-provided).
        """
        return self._ctx

    @property
    def size(self) -> tuple[int, int]:
        """Current render target size.

        Returns:
            ``(width, height)`` in pixels.
        """
        return (self._width, self._height)

    def set_colormap(self, name: str) -> None:
        """Swap the active colormap, rebuilding its GPU table if changed.

        Args:
            name: A colormap name from :func:`boidforge.viz.colormaps.available`.
        """
        if name == self._lut_name:
            return
        self._lut_tex.release()
        self._lut_tex = self._build_lut(name)
        self._lut_name = name
        self.config.colormap = name

    def resize(self, width: int, height: int) -> None:
        """Resize all render targets (e.g. on window resize).

        Args:
            width: New framebuffer width in pixels.
            height: New framebuffer height in pixels.
        """
        width = max(int(width), 1)
        height = max(int(height), 1)
        if (width, height) == (self._width, self._height):
            return
        self._release_targets()
        self._width, self._height = width, height
        self.config.width, self.config.height = width, height
        self._allocate_targets(width, height)

    def clear_trails(self) -> None:
        """Erase the accumulation buffer, removing any lingering trails."""
        self._accum_fbo.clear(0.0, 0.0, 0.0, 0.0)

    def draw(self, state: SimulationState, camera: Camera, present: bool = True) -> None:
        """Render one frame of boids with the given camera transform.

        Runs the fade, additive sprite, bloom, and composite passes. The final
        image always lands in an offscreen RGBA8 buffer (so
        :meth:`read_pixels` works for export); when ``present`` is true it is
        also blitted to the context's screen for interactive display.

        Args:
            state: Frame state to draw.
            camera: View transform for this frame.
            present: Whether to blit the result to the on-screen framebuffer.
        """
        self._upload(state)
        self._fade_accum()
        self._draw_points(state.n, camera)
        if self.config.bloom:
            self._build_bloom()
        self._composite()
        if present:
            self._ctx.copy_framebuffer(self._ctx.screen, self._out_fbo)

    def read_pixels(self) -> bytes:
        """Read the composited framebuffer as raw RGBA bytes (top-row first).

        Returns:
            Tightly packed RGBA8 pixels, row-major with the top image row
            first, length ``width*height*4`` — ready for FFmpeg.
        """
        # GL framebuffers are bottom-row first; flip to top-first for video.
        # tobytes() emits the flipped view in C order in a single copy.
        raw = self._out_fbo.read(components=4, dtype="f1")
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self._height, self._width, 4)
        return arr[::-1].tobytes()

    def release(self) -> None:
        """Release all GPU resources held by this renderer."""
        self._release_targets()
        for obj in (
            self._lut_tex,
            self._boid_vbo,
            self._boid_vao,
            self._fs_vbo,
            self._fade_vao,
            self._bright_vao,
            self._blur_vao,
            self._composite_vao,
            self._point_prog,
            self._fade_prog,
            self._bright_prog,
            self._blur_prog,
            self._composite_prog,
        ):
            obj.release()
        if self._owns_ctx:
            self._ctx.release()

    # -- internal passes ---------------------------------------------------

    def _upload(self, state: SimulationState) -> None:
        """Pack SoA state (and optional density) into the boid vertex buffer."""
        n = state.n
        if n > self._capacity:
            self._grow(n)
        s = self._scratch
        s[:n, 0] = state.px
        s[:n, 1] = state.py
        s[:n, 2] = state.vx
        s[:n, 3] = state.vy
        if self.config.color_mode is ColorMode.DENSITY:
            s[:n, 4] = self._density(state)
        else:
            s[:n, 4] = 0.0
        self._boid_vbo.write(np.ascontiguousarray(s[:n]).tobytes())

    def _density(self, state: SimulationState) -> FloatArray:
        """Per-boid local crowding, normalized to ``[0, 1]`` for the colour ramp.

        Crowding is the number of boids sharing the same uniform-grid cell of
        side :attr:`RenderConfig.density_cell`. The count is rescaled by the 5th
        and 95th percentiles of the per-boid counts so the ramp spans the actual
        spread rather than the long tail of the densest cluster.

        Args:
            state: Current frame state.

        Returns:
            A ``float32`` array of length ``N`` in ``[0, 1]``.
        """
        cell = max(self.config.density_cell, 1e-3)
        gx = np.floor(state.px.astype(np.float64) / cell).astype(np.int64)
        gy = np.floor(state.py.astype(np.float64) / cell).astype(np.int64)
        gx -= gx.min()
        gy -= gy.min()
        cell_id = gy * (gx.max() + 1) + gx
        counts = np.bincount(cell_id)
        per_boid = counts[cell_id].astype(np.float64)
        lo = float(np.percentile(per_boid, 5.0))
        hi = float(np.percentile(per_boid, 95.0))
        rng = hi - lo if hi > lo else 1.0
        normalized: FloatArray = np.clip((per_boid - lo) / rng, 0.0, 1.0).astype(DTYPE)
        return normalized

    def _grow(self, n: int) -> None:
        """Grow the vertex buffer and scratch array to hold at least ``n``."""
        self._capacity = 1 << (int(n - 1).bit_length())
        self._scratch = np.zeros((self._capacity, 5), dtype=DTYPE)
        self._boid_vao.release()
        self._boid_vbo.release()
        self._boid_vbo = self._ctx.buffer(reserve=self._capacity * 20, dynamic=True)
        self._boid_vao = self._ctx.vertex_array(
            self._point_prog,
            [(self._boid_vbo, "2f 2f 1f", "in_pos", "in_vel", "in_aux")],
        )

    def _fade_accum(self) -> None:
        """Fade the accumulation buffer toward black, or clear it if no trails."""
        decay = self.config.trail_decay
        if decay <= 0.0:
            self._accum_fbo.clear(0.0, 0.0, 0.0, 0.0)
            return
        self._accum_fbo.use()
        self._ctx.enable(self._mgl.BLEND)
        self._ctx.blend_func = (self._mgl.SRC_ALPHA, self._mgl.ONE_MINUS_SRC_ALPHA)
        self._set(self._fade_prog, "u_color", (0.0, 0.0, 0.0, 1.0 - decay))
        self._fade_vao.render(self._mgl.TRIANGLES, vertices=3)

    def _draw_points(self, n: int, camera: Camera) -> None:
        """Additively draw ``n`` boid sprites into the accumulation buffer."""
        if n <= 0:
            return
        self._accum_fbo.use()
        self._ctx.enable(self._mgl.BLEND)
        self._ctx.blend_func = (self._mgl.ONE, self._mgl.ONE)  # additive HDR

        cfg = self.config
        prog = self._point_prog
        self._set(prog, "u_mvp", camera.view_matrix(self._width, self._height))
        self._set(prog, "u_point_size", float(cfg.point_size))
        self._set(prog, "u_min_size", float(cfg.min_point_size))
        speed_range = cfg.speed_hi - cfg.speed_lo
        self._set(prog, "u_speed_lo", float(cfg.speed_lo))
        self._set(prog, "u_speed_inv_range", 1.0 / (speed_range if speed_range > 1e-3 else 1e-3))
        self._set(prog, "u_color_mode", int(cfg.color_mode))
        self._set(prog, "u_uniform_t", float(cfg.uniform_t))
        self._set(prog, "u_glow", float(cfg.glow))
        self._set(prog, "u_intensity", float(cfg.intensity))
        self._lut_tex.use(location=0)
        self._set(prog, "u_lut", 0)
        self._boid_vao.render(self._mgl.POINTS, vertices=n)

    def _build_bloom(self) -> None:
        """Bright-pass the accumulation buffer and Gaussian-blur it for glow."""
        self._ctx.disable(self._mgl.BLEND)
        bw, bh = self._bright_tex.size

        self._bright_fbo.use()
        self._accum_tex.use(location=0)
        self._set(self._bright_prog, "u_scene", 0)
        self._set(self._bright_prog, "u_threshold", float(self.config.bloom_threshold))
        self._bright_vao.render(self._mgl.TRIANGLES, vertices=3)

        # Two separable blur passes: horizontal (bright->blur0), vertical (->blur1).
        src_tex = self._bright_tex
        dirs = ((1.0 / bw, 0.0), (0.0, 1.0 / bh))
        for i, direction in enumerate(dirs):
            self._blur_fbo[i].use()
            src_tex.use(location=0)
            self._set(self._blur_prog, "u_tex", 0)
            self._set(self._blur_prog, "u_dir", direction)
            self._blur_vao.render(self._mgl.TRIANGLES, vertices=3)
            src_tex = self._blur_tex[i]

    def _composite(self) -> None:
        """Tonemap the HDR scene, add bloom, and write the RGBA8 output."""
        cfg = self.config
        self._out_fbo.use()
        self._out_fbo.clear(0.0, 0.0, 0.0, 1.0)
        self._ctx.disable(self._mgl.BLEND)

        prog = self._composite_prog
        self._accum_tex.use(location=0)
        # When bloom is off, bind the accum texture to the bloom sampler too and
        # zero its strength so the shader add is a no-op.
        (self._blur_tex[1] if cfg.bloom else self._accum_tex).use(location=1)
        self._set(prog, "u_scene", 0)
        self._set(prog, "u_bloom", 1)
        self._set(prog, "u_exposure", float(cfg.exposure))
        self._set(prog, "u_bloom_strength", float(cfg.bloom_strength) if cfg.bloom else 0.0)
        self._set(prog, "u_vignette", float(cfg.vignette))
        self._set(prog, "u_background", tuple(cfg.background))
        self._composite_vao.render(self._mgl.TRIANGLES, vertices=3)

    # -- low-level utilities ----------------------------------------------

    def _set(self, prog: moderngl.Program, name: str, value: Any) -> None:
        """Assign a uniform if present (silently skips ones GLSL optimized out)."""
        try:
            uniform = prog[name]
        except KeyError:
            return
        uniform.value = value  # type: ignore[union-attr]  # only Uniforms are set here

    def _release_targets(self) -> None:
        """Release all size-dependent framebuffers and their textures."""
        for obj in (
            getattr(self, "_accum_fbo", None),
            getattr(self, "_accum_tex", None),
            getattr(self, "_bright_fbo", None),
            getattr(self, "_bright_tex", None),
            getattr(self, "_out_fbo", None),
            getattr(self, "_out_tex", None),
        ):
            if obj is not None:
                obj.release()
        for seq in (getattr(self, "_blur_fbo", []), getattr(self, "_blur_tex", [])):
            for obj in seq:
                obj.release()
