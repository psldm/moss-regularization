// Self-test mirroring integrations/c/test_moss_damp.c.  Exit code 1 on failure.
using System;
using MossRegularization;

int nfail = 0;
void Check(bool cond, string msg) { if (!cond) { Console.Error.WriteLine("FAIL: " + msg); nfail++; } }

const double lam = 3.7, dt = 0.7;
double[] v = { 0.3, -1.2, 0.7 };
double s2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
double fRef = 1.0 / Math.Sqrt(1.0 + 2.0 * lam * s2 * dt);

double[] a = (double[])v.Clone(); MossDamp.Damp(a, lam, dt);
for (int i = 0; i < 3; i++) Check(Math.Abs(a[i] - fRef * v[i]) < 1e-15, "closed form");

double[] b = (double[])v.Clone(); MossDamp.Damp(b, lam, 0.5 * dt); MossDamp.Damp(b, lam, 0.5 * dt);
for (int i = 0; i < 3; i++) Check(Math.Abs(a[i] - b[i]) < 1e-15, "composition");

double[] c = (double[])v.Clone(); MossDamp.DampAlpha(c, lam, dt, 2.0);
for (int i = 0; i < 3; i++) Check(Math.Abs(a[i] - c[i]) < 1e-14, "alpha=2 path");

double[] arr = { 1.0, 0.0, 0.0, 0.0, 2.0, 0.0 };
double[] lams = { 1.0, 0.5 }, masses = { 2.0, 1.0 };
double eBefore = 0.5 * 2.0 * 1.0 + 0.5 * 1.0 * 4.0;
double removed = MossDamp.DampArray(arr, 3, lams, 0.0, masses, dt);
double eAfter = 0.5 * 2.0 * arr[0] * arr[0] + 0.5 * arr[4] * arr[4];
Check(Math.Abs((eBefore - eAfter) - removed) < 1e-14, "energy removed");
Check(Math.Abs(MossDamp.DtLimit(4.0, 0.5, 0.5) - 0.5) < 1e-15, "dt limit");

Console.WriteLine($"v_out={a[0]:R} {a[1]:R} {a[2]:R}");
Console.WriteLine(nfail == 0 ? "OK" : "FAILED");
return nfail == 0 ? 0 : 1;
