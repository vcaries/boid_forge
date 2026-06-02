/*
 * boids_kernel.c — core SoA boid update kernel.
 *
 * This translation unit contains the hot path: a uniform-grid neighbor search
 * and the separation/alignment/cohesion update over contiguous float buffers.
 * It is intentionally free of Python and NumPy headers so it stays a pure,
 * testable numerical kernel.
 *
 * Skeleton stage: the entry point is declared and wired, but the numerical
 * update is not yet implemented. The body is deliberately empty (it does not
 * fabricate motion); the Python wrapper surfaces NotImplementedError until the
 * kernel lands.
 */
#include "boids.h"

void bf_step(int n,
             float *px, float *py, float *vx, float *vy,
             const bf_params_t *p)
{
    /* Mark parameters as used; no-op until the kernel is implemented. */
    (void)n;
    (void)px;
    (void)py;
    (void)vx;
    (void)vy;
    (void)p;
}
