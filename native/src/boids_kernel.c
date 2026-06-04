/*
 * boids_kernel.c — core SoA boid update kernel (L4).
 *
 * The hot path: a uniform-grid neighbor search followed by the separation /
 * alignment / cohesion update over contiguous float buffers. It is free of
 * Python and NumPy headers so it stays a pure, testable numerical kernel.
 *
 * Bit-identity to the Python reference (L1/L2/L3) is the hard constraint, not an
 * afterthought. It is achieved by:
 *
 *   1. Reducing over exactly the same neighbors in ascending boid-index order.
 *      The grid only prunes pairs that can never be neighbors; whichever
 *      neighbors survive the radius test are summed in the same order L1 uses.
 *   2. Reproducing NumPy's float32 reduction *algorithm* — pairwise summation
 *      with an 8-accumulator block of size 128 (PW_BLOCKSIZE) — so the rounding
 *      of each sum/mean matches np.sum / np.mean exactly.
 *   3. Matching np.mod's float32 remainder for the wrap boundary.
 *   4. Being compiled with precise floating point (no -ffast-math / /fp:fast,
 *      no FMA contraction), so the compiler cannot reassociate these sums.
 *
 * No allocation occurs in the per-boid loop; a single workspace is sized once
 * per step.
 */
#include "boids.h"

#include <math.h>
#include <stdlib.h>

/* NumPy's pairwise-summation block size (numpy/_core/src/umath/loops_utils). */
#define PW_BLOCKSIZE 128

/*
 * float32 pairwise summation, bit-identical to NumPy's add.reduce over a
 * contiguous float32 buffer. Do not "simplify" into a plain accumulator loop:
 * the block structure is what fixes the rounding.
 */
static float
pairwise_sum(const float *a, long n)
{
    if (n < 8) {
        float res = 0.0f;
        for (long i = 0; i < n; i++) {
            res += a[i];
        }
        return res;
    }
    else if (n <= PW_BLOCKSIZE) {
        float r[8];
        long i;
        r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3];
        r[4] = a[4]; r[5] = a[5]; r[6] = a[6]; r[7] = a[7];
        for (i = 8; i < n - (n % 8); i += 8) {
            r[0] += a[i + 0]; r[1] += a[i + 1]; r[2] += a[i + 2]; r[3] += a[i + 3];
            r[4] += a[i + 4]; r[5] += a[i + 5]; r[6] += a[i + 6]; r[7] += a[i + 7];
        }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) {
            res += a[i];
        }
        return res;
    }
    else {
        long n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
    }
}

/*
 * float32 remainder matching numpy.mod (result takes the sign of the divisor).
 * For the positive world extents used here this maps any position into
 * [0, extent), exactly as np.mod(px, w) does.
 */
static float
bf_remainderf(float a, float b)
{
    float mod = fmodf(a, b);
    if (mod != 0.0f) {
        if ((b < 0.0f) != (mod < 0.0f)) {
            mod += b;
        }
    }
    else {
        mod = copysignf(0.0f, b);
    }
    return mod;
}

/* ascending integer comparator for qsort (candidate index ordering). */
static int
cmp_int(const void *a, const void *b)
{
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

/*
 * Per-step workspace. All buffers are sized to the boid count (and the grid to
 * the occupied cell range) once per step; the per-boid loop touches them but
 * never allocates.
 */
typedef struct {
    float *new_vx, *new_vy;   /* next-step velocities (synchronous update) */
    long *cell_x, *cell_y;    /* per-boid integer cell coordinates */
    long *cell_id;            /* per-boid linear cell id */
    int *cell_start;          /* CSR cell offsets, length ncell + 1 */
    int *cell_fill;           /* scatter cursor, length ncell */
    int *order;               /* boid indices grouped by cell (ascending in-cell) */
    int *cand;                /* gathered 3x3 candidate indices (per boid) */
    /* compacted neighbor buffers, one entry per surviving neighbor */
    float *sx, *sy;           /* separation terms */
    float *avx, *avy;         /* alignment neighbor velocities */
    float *cpx, *cpy;         /* cohesion neighbor positions */
} workspace_t;

static int
ws_alloc(workspace_t *w, long n, long ncell)
{
    w->new_vx = malloc((size_t)n * sizeof(float));
    w->new_vy = malloc((size_t)n * sizeof(float));
    w->cell_x = malloc((size_t)n * sizeof(long));
    w->cell_y = malloc((size_t)n * sizeof(long));
    w->cell_id = malloc((size_t)n * sizeof(long));
    w->cell_start = calloc((size_t)ncell + 1, sizeof(int));
    w->cell_fill = calloc((size_t)ncell, sizeof(int));
    w->order = malloc((size_t)n * sizeof(int));
    w->cand = malloc((size_t)n * sizeof(int));
    w->sx = malloc((size_t)n * sizeof(float));
    w->sy = malloc((size_t)n * sizeof(float));
    w->avx = malloc((size_t)n * sizeof(float));
    w->avy = malloc((size_t)n * sizeof(float));
    w->cpx = malloc((size_t)n * sizeof(float));
    w->cpy = malloc((size_t)n * sizeof(float));

    if (!w->new_vx || !w->new_vy || !w->cell_x || !w->cell_y || !w->cell_id ||
        !w->cell_start || !w->cell_fill || !w->order || !w->cand || !w->sx ||
        !w->sy || !w->avx || !w->avy || !w->cpx || !w->cpy) {
        return -1;
    }
    return 0;
}

static void
ws_free(workspace_t *w)
{
    free(w->new_vx); free(w->new_vy);
    free(w->cell_x); free(w->cell_y); free(w->cell_id);
    free(w->cell_start); free(w->cell_fill);
    free(w->order); free(w->cand);
    free(w->sx); free(w->sy);
    free(w->avx); free(w->avy);
    free(w->cpx); free(w->cpy);
}

int
bf_step(int n,
        float *px, float *py, float *vx, float *vy,
        const bf_params_t *p)
{
    if (n <= 0) {
        return 0;
    }

    const float dt = p->dt;
    const float r_sep2 = p->r_sep * p->r_sep;
    const float r_ali2 = p->r_ali * p->r_ali;
    const float r_coh2 = p->r_coh * p->r_coh;
    const float w_sep = p->w_sep;
    const float w_ali = p->w_ali;
    const float w_coh = p->w_coh;
    const float max_force = p->max_force;
    const float max_force2 = p->max_force * p->max_force;
    const float max_speed = p->max_speed;
    const float min_speed = p->min_speed;

    /*
     * Cell side = largest interaction radius, matching the L3 grid. Binning is
     * done in double purely to choose buckets; it never enters the physics, so
     * it cannot perturb results — it only must not drop a true neighbor, which
     * a cell side >= every radius guarantees within the 3x3 block.
     */
    double s = (double)p->r_sep;
    if ((double)p->r_ali > s) s = (double)p->r_ali;
    if ((double)p->r_coh > s) s = (double)p->r_coh;

    /* Pass 1: cell coordinates and occupied-range bounds. */
    long min_cx = 0, max_cx = 0, min_cy = 0, max_cy = 0;
    long *cx_tmp = malloc((size_t)n * sizeof(long));
    long *cy_tmp = malloc((size_t)n * sizeof(long));
    if (!cx_tmp || !cy_tmp) {
        free(cx_tmp); free(cy_tmp);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        long cx = (long)floor((double)px[i] / s);
        long cy = (long)floor((double)py[i] / s);
        cx_tmp[i] = cx;
        cy_tmp[i] = cy;
        if (i == 0) {
            min_cx = max_cx = cx;
            min_cy = max_cy = cy;
        }
        else {
            if (cx < min_cx) min_cx = cx;
            if (cx > max_cx) max_cx = cx;
            if (cy < min_cy) min_cy = cy;
            if (cy > max_cy) max_cy = cy;
        }
    }
    const long ncx = max_cx - min_cx + 1;
    const long ncy = max_cy - min_cy + 1;
    const long ncell = ncx * ncy;

    workspace_t w;
    if (ws_alloc(&w, n, ncell) != 0) {
        ws_free(&w);
        free(cx_tmp); free(cy_tmp);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        w.cell_x[i] = cx_tmp[i];
        w.cell_y[i] = cy_tmp[i];
        w.cell_id[i] = (cy_tmp[i] - min_cy) * ncx + (cx_tmp[i] - min_cx);
    }
    free(cx_tmp);
    free(cy_tmp);

    /* Counting sort of boid indices into cells (CSR). Scanning i ascending
     * leaves each cell's slice in ascending boid-index order. */
    for (int i = 0; i < n; i++) {
        w.cell_start[w.cell_id[i] + 1]++;
    }
    for (long c = 1; c <= ncell; c++) {
        w.cell_start[c] += w.cell_start[c - 1];
    }
    for (int i = 0; i < n; i++) {
        long c = w.cell_id[i];
        w.order[w.cell_start[c] + w.cell_fill[c]] = i;
        w.cell_fill[c]++;
    }

    /* Per-boid update. Reads only start-of-step state; writes new_v*. */
    for (int i = 0; i < n; i++) {
        const long cx = w.cell_x[i];
        const long cy = w.cell_y[i];
        const float pxi = px[i];
        const float pyi = py[i];

        /* Gather the 3x3 cell block, then sort to ascending global index so the
         * surviving neighbors reduce in L1's order. */
        int m = 0;
        for (long gx = cx - 1; gx <= cx + 1; gx++) {
            if (gx < min_cx || gx > max_cx) continue;
            for (long gy = cy - 1; gy <= cy + 1; gy++) {
                if (gy < min_cy || gy > max_cy) continue;
                long c = (gy - min_cy) * ncx + (gx - min_cx);
                for (int k = w.cell_start[c]; k < w.cell_start[c + 1]; k++) {
                    w.cand[m++] = w.order[k];
                }
            }
        }
        qsort(w.cand, (size_t)m, sizeof(int), cmp_int);

        int cs = 0, ca = 0, cc = 0;
        for (int k = 0; k < m; k++) {
            int j = w.cand[k];
            if (j == i) {
                continue;
            }
            float dx = px[j] - pxi;   /* neighbor minus self, as in L1 */
            float dy = py[j] - pyi;
            float dist2 = dx * dx + dy * dy;
            if (dist2 < r_sep2) {
                float inv = 1.0f / dist2;
                w.sx[cs] = (-dx) * inv;
                w.sy[cs] = (-dy) * inv;
                cs++;
            }
            if (dist2 < r_ali2) {
                w.avx[ca] = vx[j];
                w.avy[ca] = vy[j];
                ca++;
            }
            if (dist2 < r_coh2) {
                w.cpx[cc] = px[j];
                w.cpy[cc] = py[j];
                cc++;
            }
        }

        float ax = 0.0f;
        float ay = 0.0f;
        if (cs > 0) {
            ax = ax + w_sep * pairwise_sum(w.sx, cs);
            ay = ay + w_sep * pairwise_sum(w.sy, cs);
        }
        if (ca > 0) {
            float mvx = pairwise_sum(w.avx, ca) / (float)ca;
            float mvy = pairwise_sum(w.avy, ca) / (float)ca;
            ax = ax + w_ali * (mvx - vx[i]);
            ay = ay + w_ali * (mvy - vy[i]);
        }
        if (cc > 0) {
            float mpx = pairwise_sum(w.cpx, cc) / (float)cc;
            float mpy = pairwise_sum(w.cpy, cc) / (float)cc;
            ax = ax + w_coh * (mpx - pxi);
            ay = ay + w_coh * (mpy - pyi);
        }

        float a2 = ax * ax + ay * ay;
        if (a2 > max_force2) {
            float scale = max_force / sqrtf(a2);
            ax = ax * scale;
            ay = ay * scale;
        }

        float nvx = vx[i] + ax * dt;
        float nvy = vy[i] + ay * dt;

        float speed = sqrtf(nvx * nvx + nvy * nvy);
        if (speed > max_speed) {
            float scale = max_speed / speed;
            nvx = nvx * scale;
            nvy = nvy * scale;
        }
        else if (speed < min_speed && speed > 0.0f) {
            float scale = min_speed / speed;
            nvx = nvx * scale;
            nvy = nvy * scale;
        }

        w.new_vx[i] = nvx;
        w.new_vy[i] = nvy;
    }

    /* Commit velocities, then integrate positions with the new velocities, then
     * apply the boundary — exactly L1's ordering. */
    for (int i = 0; i < n; i++) {
        vx[i] = w.new_vx[i];
        vy[i] = w.new_vy[i];
    }
    for (int i = 0; i < n; i++) {
        px[i] = px[i] + vx[i] * dt;
        py[i] = py[i] + vy[i] * dt;
    }

    const float world_w = p->world_w;
    const float world_h = p->world_h;
    if (p->boundary == BF_BOUNDARY_WRAP) {
        for (int i = 0; i < n; i++) {
            px[i] = bf_remainderf(px[i], world_w);
            py[i] = bf_remainderf(py[i], world_h);
        }
    }
    else { /* BF_BOUNDARY_REFLECT */
        for (int i = 0; i < n; i++) {
            if (px[i] < 0.0f) {
                px[i] = -px[i];
                vx[i] = -vx[i];
            }
            if (px[i] > world_w) {
                px[i] = 2.0f * world_w - px[i];
                vx[i] = -vx[i];
            }
            if (py[i] < 0.0f) {
                py[i] = -py[i];
                vy[i] = -vy[i];
            }
            if (py[i] > world_h) {
                py[i] = 2.0f * world_h - py[i];
                vy[i] = -vy[i];
            }
        }
    }

    ws_free(&w);
    return 0;
}
