/*
 * moss_damp.h  --  header-only exact substep of the vacuum-damping ODE
 *
 *     dv/dt = -lam |v|^alpha v          (alpha = 2: cubic damping)
 *
 * whose solution over a step dt preserves the direction of v and rescales
 * its magnitude:
 *
 *     alpha = 2 :  v <- v / sqrt(1 + 2 lam |v|^2 dt)
 *     general   :  v <- v * (1 + alpha lam |v|^alpha dt)^(-1/alpha)
 *
 * The step is exact for any dt (no stability limit) and the exact flow
 * composes: two steps of dt/2 equal one step of dt to round-off.  Use it
 * as the damping substep of an operator splitting: predictor for
 * convection / viscosity / gravity, then moss_damp() on the velocity.
 *
 * lam is the effective coupling in the units of your solver
 * (lam = G rho / c^3 in code units, or a dimensionless damping number
 * lam_code = lam_eff U^alpha L; see the moss-reg README for what that
 * value does and does not mean physically).
 *
 * C99, C++-compatible, no allocations, no dependencies beyond <math.h>.
 * Link with -lm.  MIT license (moss-regularization).
 */
#ifndef MOSS_DAMP_H
#define MOSS_DAMP_H

#include <math.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MOSS_DAMP_VERSION "0.3.0"

/* Magnitude factor f such that v_new = f * v, cubic damping. */
static inline double moss_damp_factor(double speed2, double lam, double dt)
{
    return 1.0 / sqrt(1.0 + 2.0 * lam * speed2 * dt);
}

/* Magnitude factor for a general exponent alpha >= 0 (alpha = 2 is cubic). */
static inline double moss_damp_factor_alpha(double speed2, double lam, double dt, double alpha)
{
    if (speed2 <= 0.0) return 1.0;
    return pow(1.0 + alpha * lam * pow(speed2, 0.5 * alpha) * dt, -1.0 / alpha);
}

/* In-place exact cubic damping of one velocity vector v[0..ndim-1]. */
static inline void moss_damp(double *v, int ndim, double lam, double dt)
{
    double s2 = 0.0, f;
    int i;
    for (i = 0; i < ndim; ++i) s2 += v[i] * v[i];
    f = moss_damp_factor(s2, lam, dt);
    for (i = 0; i < ndim; ++i) v[i] *= f;
}

/* In-place exact damping with a general exponent. */
static inline void moss_damp_alpha(double *v, int ndim, double lam, double dt, double alpha)
{
    double s2 = 0.0, f;
    int i;
    for (i = 0; i < ndim; ++i) s2 += v[i] * v[i];
    f = moss_damp_factor_alpha(s2, lam, dt, alpha);
    for (i = 0; i < ndim; ++i) v[i] *= f;
}

/*
 * Damp n velocity vectors stored contiguously (v[k*ndim + i]).
 * lam_per_item may be NULL (then lam0 is used for every item), or point to
 * n per-item couplings (e.g. lam_i = G rho_i / c^3 from a local density).
 * Returns the total kinetic energy removed, sum_k 1/2 m_k (|v|^2 - |v_new|^2),
 * with masses from mass_per_item (NULL -> unit masses); accumulate it as the
 * energy handed to the vacuum reservoir E_vac.
 */
static inline double moss_damp_array(double *v, size_t n, int ndim,
                                     const double *lam_per_item, double lam0,
                                     const double *mass_per_item, double dt)
{
    double removed = 0.0;
    size_t k;
    for (k = 0; k < n; ++k) {
        double *vk = v + k * (size_t)ndim;
        double s2 = 0.0, f, lam, m;
        int i;
        for (i = 0; i < ndim; ++i) s2 += vk[i] * vk[i];
        lam = lam_per_item ? lam_per_item[k] : lam0;
        m = mass_per_item ? mass_per_item[k] : 1.0;
        f = moss_damp_factor(s2, lam, dt);
        for (i = 0; i < ndim; ++i) vk[i] *= f;
        removed += 0.5 * m * s2 * (1.0 - f * f);
    }
    return removed;
}

/*
 * Accuracy-limited timestep for *explicit* treatments of the damping term
 * (Euler / RK stages): dt <= safety * 2 / (lam |v|_max^2).  The exact
 * substep above does not need it; use it for splitting accuracy or when
 * the term is evaluated inside an explicit RHS.
 */
static inline double moss_damp_dt_limit(double speed2_max, double lam, double safety)
{
    if (lam <= 0.0 || speed2_max <= 0.0) return HUGE_VAL;
    return safety * 2.0 / (lam * speed2_max);
}

/* Single-precision convenience variant. */
static inline void moss_damp_f(float *v, int ndim, float lam, float dt)
{
    float s2 = 0.0f, f;
    int i;
    for (i = 0; i < ndim; ++i) s2 += v[i] * v[i];
    f = 1.0f / sqrtf(1.0f + 2.0f * lam * s2 * dt);
    for (i = 0; i < ndim; ++i) v[i] *= f;
}

#ifdef __cplusplus
}
#endif

#endif /* MOSS_DAMP_H */
