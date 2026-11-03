#ifndef WELVET_MHA_COMMON_INC
#define WELVET_MHA_COMMON_INC

#include "welvet_f64math.inc"

bool mha_attn_allow(int q_pos, int k_pos, int mask, int window, int msl, int prefix_len,
                    int window_causal) {
  if (mask == 2) return true;
  if (mask == 3) {
    int w = window > 0 ? window : msl;
    if (window_causal != 0) return k_pos <= q_pos && k_pos >= q_pos - w + 1;
    int d = q_pos - k_pos;
    if (d < 0) d = -d;
    return d < w;
  }
  if (mask == 4) {
    if (q_pos < prefix_len && k_pos < prefix_len) return true;
    return k_pos <= q_pos;
  }
  return k_pos <= q_pos;
}

#define MHA_APPLY_ROPE(arr, base, pos, num_heads, head_dim, theta)                                   \
  do {                                                                                                \
    int _half = (head_dim) / 2;                                                                       \
    if (_half > 0 && (theta) > 0.0) {                                                                  \
      for (int _h = 0; _h < (num_heads); ++_h) {                                                      \
        int _hbase = (base) + _h * (head_dim);                                                        \
        for (int _d = 0; _d < _half; ++_d) {                                                          \
          double _angle =                                                                             \
              double(pos) / welvet_d_pow((theta), double(2 * _d) / double(head_dim));                 \
          double _c = welvet_d_cos(_angle);                                                           \
          double _s = welvet_d_sin(_angle);                                                           \
          double _v0 = (arr)[_hbase + _d];                                                            \
          double _v1 = (arr)[_hbase + _d + _half];                                                    \
          (arr)[_hbase + _d] = _v0 * _c - _v1 * _s;                                                   \
          (arr)[_hbase + _d + _half] = _v0 * _s + _v1 * _c;                                           \
        }                                                                                             \
      }                                                                                               \
    }                                                                                                 \
  } while (false)

#define MHA_APPLY_ROPE_BWD(arr, base, pos, num_heads, head_dim, theta)                              \
  do {                                                                                                \
    int _half = (head_dim) / 2;                                                                       \
    if (_half > 0 && (theta) > 0.0) {                                                                  \
      for (int _h = 0; _h < (num_heads); ++_h) {                                                      \
        int _hbase = (base) + _h * (head_dim);                                                        \
        for (int _d = 0; _d < _half; ++_d) {                                                          \
          double _angle =                                                                             \
              double(pos) / welvet_d_pow((theta), double(2 * _d) / double(head_dim));                 \
          double _c = welvet_d_cos(_angle);                                                           \
          double _s = welvet_d_sin(_angle);                                                           \
          double _v0 = (arr)[_hbase + _d];                                                            \
          double _v1 = (arr)[_hbase + _d + _half];                                                    \
          (arr)[_hbase + _d] = _v0 * _c + _v1 * _s;                                                   \
          (arr)[_hbase + _d + _half] = -_v0 * _s + _v1 * _c;                                          \
        }                                                                                             \
      }                                                                                               \
    }                                                                                                 \
  } while (false)

double mha_alibi_bias(int head, int num_heads, int q_pos, int k_pos, int use_alibi) {
  if (use_alibi == 0) return 0.0;
  int n = num_heads;
  if (n <= 0) n = 1;
  double slope = welvet_d_pow(2.0, -double(head + 1) * 8.0 / double(n));
  int diff = q_pos - k_pos;
  return -slope * double(diff < 0 ? -diff : diff);
}

#endif
