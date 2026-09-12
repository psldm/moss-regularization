// MossDamp.cs -- exact substep of dv/dt = -lam |v|^alpha v for C# / .NET.
// Mirror of integrations/c/moss_damp.h.  MIT license (moss-regularization).
using System;

namespace MossRegularization
{
    /// <summary>Exact damping substep, direction preserving, no stability limit.</summary>
    public static class MossDamp
    {
        public const string Version = "0.4.0";

        /// <summary>Magnitude factor f with v_new = f v, cubic damping (alpha = 2).</summary>
        public static double Factor(double speed2, double lam, double dt)
            => 1.0 / Math.Sqrt(1.0 + 2.0 * lam * speed2 * dt);

        /// <summary>Magnitude factor for a general exponent alpha.</summary>
        public static double FactorAlpha(double speed2, double lam, double dt, double alpha)
            => speed2 <= 0.0 ? 1.0 : Math.Pow(1.0 + alpha * lam * Math.Pow(speed2, 0.5 * alpha) * dt, -1.0 / alpha);

        /// <summary>In-place exact cubic damping of one velocity vector.</summary>
        public static void Damp(Span<double> v, double lam, double dt)
        {
            double s2 = 0.0;
            for (int i = 0; i < v.Length; i++) s2 += v[i] * v[i];
            double f = Factor(s2, lam, dt);
            for (int i = 0; i < v.Length; i++) v[i] *= f;
        }

        /// <summary>In-place exact damping with a general exponent.</summary>
        public static void DampAlpha(Span<double> v, double lam, double dt, double alpha)
        {
            double s2 = 0.0;
            for (int i = 0; i < v.Length; i++) s2 += v[i] * v[i];
            double f = FactorAlpha(s2, lam, dt, alpha);
            for (int i = 0; i < v.Length; i++) v[i] *= f;
        }

        /// <summary>
        /// Damp n vectors stored contiguously (v[k*ndim + i]); lamPerItem / massPerItem may be null.
        /// Returns the kinetic energy removed (accumulate as E_vac).
        /// </summary>
        public static double DampArray(Span<double> v, int ndim, ReadOnlySpan<double> lamPerItem, double lam0,
                                       ReadOnlySpan<double> massPerItem, double dt)
        {
            int n = v.Length / ndim;
            double removed = 0.0;
            for (int k = 0; k < n; k++)
            {
                var vk = v.Slice(k * ndim, ndim);
                double s2 = 0.0;
                for (int i = 0; i < ndim; i++) s2 += vk[i] * vk[i];
                double lam = lamPerItem.IsEmpty ? lam0 : lamPerItem[k];
                double m = massPerItem.IsEmpty ? 1.0 : massPerItem[k];
                double f = Factor(s2, lam, dt);
                for (int i = 0; i < ndim; i++) vk[i] *= f;
                removed += 0.5 * m * s2 * (1.0 - f * f);
            }
            return removed;
        }

        /// <summary>Accuracy-limited dt for explicit treatments: safety * 2 / (lam |v|max^2).</summary>
        public static double DtLimit(double speed2Max, double lam, double safety)
            => (lam <= 0.0 || speed2Max <= 0.0) ? double.PositiveInfinity : safety * 2.0 / (lam * speed2Max);
    }
}
