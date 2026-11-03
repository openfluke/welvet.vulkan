#ifndef WELVET_SOFTDOUBLE_INC
#define WELVET_SOFTDOUBLE_INC

// Soft-double (vec2 hi/lo) — port of welvet.metal ops_common / welvet_dense.
// Do NOT enable GL_ARB_gpu_shader_fp64 here: that extension breaks float
// log/exp/ldexp on glslang; parents that need doubles enable it themselves.
// Bit-exact ldexp/round avoid relying on possibly-broken float builtins.

float sd_ldexp(float x, int n) {
  if (x == 0.0 || n == 0) return x;
  uint u = floatBitsToUint(x);
  int e = int((u >> 23) & 0xFFu);
  if (e == 0xFF) return x; // inf/nan
  if (e == 0) {
    x *= 8388608.0; // 2^23 — lift subnormal
    n -= 23;
    u = floatBitsToUint(x);
    e = int((u >> 23) & 0xFFu);
    if (e == 0) return 0.0;
  }
  e += n;
  if (e >= 0xFF) {
    u = (u & 0x80000000u) | 0x7F800000u;
    return uintBitsToFloat(u);
  }
  if (e <= 0) {
    // Flush underflow (Metal ldexp similarly yields 0 for large negative n).
    return ((u & 0x80000000u) != 0u) ? -0.0 : 0.0;
  }
  u = (u & 0x807FFFFFu) | (uint(e) << 23);
  return uintBitsToFloat(u);
}

// Round half away from zero for range reduction (stable vs broken round()).
int sd_iround(float x) {
  return x >= 0.0 ? int(floor(x + 0.5)) : int(ceil(x - 0.5));
}

vec2 sd_two_sum(float a, float b) {
  float s = a + b;
  float v = s - a;
  float e = (a - (s - v)) + (b - v);
  return vec2(s, e);
}

vec2 sd_fast_two_sum(float a, float b) {
  float s = a + b;
  float e = b - (s - a);
  return vec2(s, e);
}

vec2 sd_add(vec2 a, vec2 b) {
  vec2 s = sd_two_sum(a.x, b.x);
  float t = a.y + b.y;
  return sd_two_sum(s.x, t + s.y);
}

vec2 sd_mul_float(vec2 a, float c) {
  float p = a.x * c;
  float e = fma(a.x, c, -p); // exact residual of a.x*c when fma is IEEE
  e = e + a.y * c;
  return sd_fast_two_sum(p, e);
}

vec2 sd_mul(vec2 a, vec2 b) { return sd_add(sd_mul_float(a, b.x), sd_mul_float(a, b.y)); }

float sd_to_float(vec2 a) { return a.x + a.y; }

vec2 sd_from_f64(double x) {
  float hi = float(x);
  return vec2(hi, float(x - double(hi)));
}

double sd_to_f64(vec2 a) { return double(a.x) + double(a.y); }

vec2 sd_div_uint(vec2 a, uint n) {
  vec2 d = sd_two_sum(float(n & 0xffff0000u), float(n & 0xffffu));
  float q = a.x / d.x;
  vec2 residual = sd_add(a, -sd_mul_float(d, q));
  return sd_add(vec2(q, 0.0), vec2((residual.x + residual.y) / d.x, 0.0));
}

vec2 sd_div(vec2 x, vec2 y) {
  float q = x.x / y.x;
  return sd_add(vec2(q, 0.0),
                vec2(sd_to_float(sd_add(x, -sd_mul_float(y, q))) / y.x, 0.0));
}

vec2 sd_sqrt(vec2 x) {
  float q = sqrt(x.x);
  if (q == 0.0) return vec2(0.0);
  return sd_add(vec2(q, 0.0),
                sd_div(sd_add(x, -sd_mul_float(vec2(q, 0.0), q)), vec2(2.0 * q, 0.0)));
}

vec2 sd_exp(vec2 x) {
  if (x.x > 88.0) return vec2(uintBitsToFloat(0x7F800000u), 0.0);
  if (x.x < -100.0) return vec2(0.0);
  int k = sd_iround(x.x * 1.4426950408889634);
  vec2 ln2 = vec2(0.6931471824645996, -1.904654323148236e-9);
  vec2 z = sd_add(x, -sd_mul_float(ln2, float(k)));
  vec2 term = vec2(1.0, 0.0);
  vec2 sum = term;
  for (int i = 1; i <= 18; ++i) {
    term = sd_div_uint(sd_mul(term, z), uint(i));
    sum = sd_add(sum, term);
  }
  return vec2(sd_ldexp(sum.x, k), sd_ldexp(sum.y, k));
}

vec2 sd_sigmoid(vec2 x) { return sd_div(vec2(1.0, 0.0), sd_add(vec2(1.0, 0.0), sd_exp(-x))); }

vec2 sd_tanh(vec2 x) {
  if (x.x > 18.0) return vec2(1.0, 0.0);
  if (x.x < -18.0) return vec2(-1.0, 0.0);
  vec2 ex = sd_exp(sd_mul_float(x, 2.0));
  return sd_div(sd_add(ex, vec2(-1.0, 0.0)), sd_add(ex, vec2(1.0, 0.0)));
}

// Seed with log2 (float log broken under fp64 ext), then Newton via sd_exp.
vec2 sd_log(vec2 x) {
  if (x.x <= 0.0) return vec2(log2(max(x.x, 0.0)) * 0.6931471824645996, 0.0);
  float q = log2(x.x) * 0.6931471824645996;
  vec2 ex = sd_exp(vec2(q, 0.0));
  return sd_add(vec2(q, 0.0), sd_div(sd_add(x, -ex), ex));
}

vec2 sd_sin(vec2 x) {
  vec2 pi2 = vec2(6.2831854820251465, -1.7484555314695172e-7);
  float k = float(sd_iround(x.x / 6.283185307179586));
  x = sd_add(x, -sd_mul_float(pi2, k));
  vec2 term = x;
  vec2 sum = x;
  vec2 xx = sd_mul(x, x);
  for (int n = 1; n <= 15; ++n) {
    term = -sd_div_uint(sd_mul(term, xx), uint((2 * n) * (2 * n + 1)));
    sum = sd_add(sum, term);
  }
  return sum;
}

vec2 sd_cos(vec2 x) {
  return sd_sin(sd_add(x, vec2(1.5707963705062866, -4.371138828673793e-8)));
}

#endif
