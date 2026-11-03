#ifndef WELVET_F64MATH_INC
#define WELVET_F64MATH_INC

#extension GL_ARB_gpu_shader_fp64 : enable

// True f64 transcendental polys — Vulkan fp64 ALU has +*/sqrt but no
// exp/sin/cos/log builtins under glslang. Match Go math.Exp/Sin/Cos.

double welvet_ldexp_d(double s, int n) {
  while (n >= 64) {
    s *= 18446744073709551616.0;
    n -= 64;
  }
  while (n <= -64) {
    s *= 5.4210108624275221700372640043497e-20;
    n += 64;
  }
  while (n >= 16) {
    s *= 65536.0;
    n -= 16;
  }
  while (n <= -16) {
    s *= 1.52587890625e-5;
    n += 16;
  }
  while (n >= 4) {
    s *= 16.0;
    n -= 4;
  }
  while (n <= -4) {
    s *= 0.0625;
    n += 4;
  }
  while (n > 0) {
    s *= 2.0;
    --n;
  }
  while (n < 0) {
    s *= 0.5;
    ++n;
  }
  return s;
}

// Half away from zero — matches C/Go libm. GLSL round() is round-even.
int welvet_iround(double x) {
  return int(x >= 0.0 ? floor(x + 0.5) : ceil(x - 0.5));
}

double welvet_d_exp(double x) {
  if (x != x) return x;
  if (x > 709.782712893384) return 1.0 / 0.0;
  if (x < -745.1332191019411) return 0.0;
  // fdlibm-style constants
  const double inv_ln2 = 1.442695040888963387;
  const double ln2_hi = 0.693147180369123816729;
  const double ln2_lo = 1.90821492927058770002e-10;
  int n = welvet_iround(x * inv_ln2);
  double z = (x - double(n) * ln2_hi) - double(n) * ln2_lo;
  // Pade / Taylor for exp(z), |z|≤ln2/2
  double z2 = z * z;
  // exp(z) ≈ 1 + z + z^2 * (1/2 + z/6 + ...)
  double sum = 1.0 + z;
  double term = z;
  for (int i = 2; i <= 24; ++i) {
    term *= z / double(i);
    sum += term;
  }
  // suppress unused if compiler warns
  z2 = z2;
  return welvet_ldexp_d(sum, n);
}

double welvet_d_log(double x) {
  if (!(x > 0.0)) {
    if (x == 0.0) return -1.0 / 0.0;
    return 0.0 / 0.0;
  }
  double q = double(log2(float(x))) * 0.69314718055994530941723212145818;
  for (int i = 0; i < 5; ++i) {
    double ex = welvet_d_exp(q);
    q = q + (x - ex) / ex;
  }
  return q;
}

double welvet_d_sin(double x) {
  if (x != x) return x;
  const double twopi_hi = 6.283185307179586476925;
  const double twopi_lo = 2.449293598294706414e-16;
  double k = double(welvet_iround(x / twopi_hi));
  x = (x - k * twopi_hi) - k * twopi_lo;
  double x2 = x * x;
  double term = x;
  double sum = x;
  for (int n = 1; n <= 18; ++n) {
    term *= -x2 / double((2 * n) * (2 * n + 1));
    sum += term;
  }
  return sum;
}

double welvet_d_cos(double x) {
  const double pio2_hi = 1.570796326794896619231;
  const double pio2_lo = 6.123233995736766e-17;
  return welvet_d_sin((x + pio2_hi) + pio2_lo);
}

double welvet_d_pow(double a, double b) {
  if (b == 0.0) return 1.0;
  if (a == 0.0) return 0.0;
  if (a == 2.0) return welvet_d_exp(b * 0.69314718055994530941723212145818);
  if (!(a > 0.0)) return double(pow(float(a), float(b)));
  return welvet_d_exp(b * welvet_d_log(a));
}

double welvet_d_tanh(double x) {
  if (x > 18.0) return 1.0;
  if (x < -18.0) return -1.0;
  double e2 = welvet_d_exp(2.0 * x);
  return (e2 - 1.0) / (e2 + 1.0);
}

double welvet_d_sigmoid(double x) { return 1.0 / (1.0 + welvet_d_exp(-x)); }

#endif
