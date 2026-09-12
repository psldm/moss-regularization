/* Self-test of moss_damp.h.  Prints reference numbers that the Python test
 * suite compares with moss_reg.core.analytical.exact_damping. */
#include <stdio.h>
#include <stdlib.h>
#include "moss_damp.h"

static int nfail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); nfail++; } } while (0)

int main(void)
{
    const double lam = 3.7, dt = 0.7;
    double v[3] = {0.3, -1.2, 0.7};
    double s2 = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    double f_ref = 1.0 / sqrt(1.0 + 2.0 * lam * s2 * dt);
    double a[3] = {0.3, -1.2, 0.7}, b[3] = {0.3, -1.2, 0.7}, c[3] = {0.3, -1.2, 0.7};
    double mass = 2.0, removed, arr[6] = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0}, lams[2] = {1.0, 0.5};
    int i;

    /* closed form */
    moss_damp(a, 3, lam, dt);
    for (i = 0; i < 3; ++i) CHECK(fabs(a[i] - f_ref * v[i]) < 1e-15, "closed form");

    /* composition: two half steps == one full step */
    moss_damp(b, 3, lam, 0.5 * dt);
    moss_damp(b, 3, lam, 0.5 * dt);
    for (i = 0; i < 3; ++i) CHECK(fabs(a[i] - b[i]) < 1e-15, "composition");

    /* general alpha with alpha = 2 reproduces the cubic path */
    moss_damp_alpha(c, 3, lam, dt, 2.0);
    for (i = 0; i < 3; ++i) CHECK(fabs(a[i] - c[i]) < 1e-14, "alpha=2 path");

    /* direction preserved */
    CHECK(fabs(a[0] * v[1] - a[1] * v[0]) < 1e-15 && fabs(a[1] * v[2] - a[2] * v[1]) < 1e-15, "direction");

    /* array variant with per-item lam and masses, energy bookkeeping */
    {
        double masses[2] = {mass, 1.0};
        double e_before = 0.5 * mass * 1.0 + 0.5 * 1.0 * 4.0, e_after;
        removed = moss_damp_array(arr, 2, 3, lams, 0.0, masses, dt);
        e_after = 0.5 * mass * (arr[0]*arr[0]) + 0.5 * (arr[4]*arr[4]);
        CHECK(fabs((e_before - e_after) - removed) < 1e-14, "energy removed");
        CHECK(fabs(arr[0] - 1.0 / sqrt(1.0 + 2.0 * 1.0 * 1.0 * dt)) < 1e-15, "array item 0");
        CHECK(fabs(arr[4] - 2.0 / sqrt(1.0 + 2.0 * 0.5 * 4.0 * dt)) < 1e-15, "array item 1");
    }

    /* dt limit */
    CHECK(fabs(moss_damp_dt_limit(4.0, 0.5, 0.5) - 0.5) < 1e-15, "dt limit");
    CHECK(moss_damp_dt_limit(0.0, 0.5, 0.5) == HUGE_VAL, "dt limit zero speed");

    /* reference output for the Python cross-check */
    printf("lam=%.17g dt=%.17g\n", lam, dt);
    printf("v_in=%.17g %.17g %.17g\n", v[0], v[1], v[2]);
    printf("v_out=%.17g %.17g %.17g\n", a[0], a[1], a[2]);
    {
        double d[3] = {0.3, -1.2, 0.7};
        moss_damp_alpha(d, 3, lam, dt, 3.0);
        printf("v_out_alpha3=%.17g %.17g %.17g\n", d[0], d[1], d[2]);
    }
    printf("%s\n", nfail ? "FAILED" : "OK");
    return nfail ? 1 : 0;
}
