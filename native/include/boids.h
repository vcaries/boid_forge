/*
 * boids.h — public interface of the BoidForge C kernel.
 *
 * The kernel operates exclusively on contiguous Struct-of-Arrays buffers
 * (px, py, vx, vy), each `float`, length `n`, owned by the caller (NumPy).
 * It performs no allocation in the step loop, no I/O, and no rendering.
 *
 * Determinism: neighbor contributions are accumulated in ascending boid-index
 * order so results are bit-identical to the Python reference backend.
 */
#ifndef BOIDFORGE_BOIDS_H
#define BOIDFORGE_BOIDS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Boundary handling modes (mirror boidforge.core.config.BoundaryMode). */
typedef enum {
    BF_BOUNDARY_WRAP = 0,
    BF_BOUNDARY_REFLECT = 1
} bf_boundary_t;

/*
 * Simulation parameters for a single step. All distances/speeds are in world
 * units; weights are dimensionless. Mirrors the relevant fields of
 * boidforge.core.config.SimulationConfig.
 */
typedef struct {
    float dt;

    float r_sep; /* separation radius */
    float r_ali; /* alignment radius  */
    float r_coh; /* cohesion radius   */

    float w_sep; /* separation weight */
    float w_ali; /* alignment weight  */
    float w_coh; /* cohesion weight   */

    float max_speed;
    float min_speed;
    float max_force;

    float world_w;
    float world_h;

    int boundary; /* bf_boundary_t */
} bf_params_t;

/*
 * Advance the simulation by one timestep, mutating the SoA buffers in place.
 *
 * Parameters:
 *   n   : number of boids (length of each array).
 *   px,py,vx,vy : contiguous float arrays of length n (positions, velocities).
 *   p   : simulation parameters (non-NULL).
 *
 * The caller guarantees the four arrays are non-aliasing, contiguous, and of
 * length n. Uses an internal uniform grid for ~O(n) neighbor queries.
 *
 * Determinism: the neighbor reductions reproduce NumPy's float32 pairwise
 * summation over the same neighbors taken in ascending boid-index order, so the
 * output is bit-identical to the L1/L2/L3 Python backends. This requires the
 * translation unit to be compiled with precise (non-reassociating, non-FMA-
 * contracting) floating point — see CMakeLists.txt.
 *
 * Returns 0 on success, or -1 if a transient workspace allocation failed (in
 * which case the buffers are left unmodified). No allocation occurs in the
 * per-boid hot loop.
 */
int bf_step(int n,
            float *px, float *py, float *vx, float *vy,
            const bf_params_t *p);

#ifdef __cplusplus
}
#endif

#endif /* BOIDFORGE_BOIDS_H */
