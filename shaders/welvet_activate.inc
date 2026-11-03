#ifndef WELVET_ACTIVATE_INC
#define WELVET_ACTIVATE_INC

#include "welvet_f64math.inc"

double welvet_activate_dev(int act, double x) {
  switch (act) {
  case -1:
    return x;
  case 0:
    return x < 0.0 ? 0.0 : x;
  case 6:
    return x < 0.0 ? 0.0 : x * x;
  case 5:
    return x < 0.0 ? 0.01 * x : x;
  case 1: {
    // Only call exp on ≤0 (same domain Softmax validates). For x<0 use
    // x*e^x/(1+e^x) so exp arg is x≤0; for x≥0 use x/(1+e^{-x}).
    if (x >= 0.0) return x / (1.0 + welvet_d_exp(-x));
    double e = welvet_d_exp(x);
    return x * e / (1.0 + e);
  }
  case 2: {
    const double k = 0.7978845608028654;
    return 0.5 * x * (1.0 + welvet_d_tanh(k * (x + 0.044715 * x * x * x)));
  }
  case 3:
    return welvet_d_tanh(x);
  case 4:
    return welvet_d_sigmoid(x);
  default:
    return x;
  }
}

double welvet_activate_deriv_dev(int act, double x) {
  switch (act) {
  case -1:
    return 1.0;
  case 0:
    return x > 0.0 ? 1.0 : 0.0;
  case 6:
    return x > 0.0 ? 2.0 * x : 0.0;
  case 5:
    return x > 0.0 ? 1.0 : 0.01;
  case 1: {
    const double sig = welvet_d_sigmoid(x);
    return sig * (1.0 + x * (1.0 - sig));
  }
  case 2: {
    const double k = 0.7978845608028654;
    const double u = k * (x + 0.044715 * x * x * x);
    const double th = welvet_d_tanh(u);
    const double du = k * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du;
  }
  case 3: {
    const double t = welvet_d_tanh(x);
    return 1.0 - t * t;
  }
  case 4: {
    const double s = welvet_d_sigmoid(x);
    return s * (1.0 - s);
  }
  default:
    return 1.0;
  }
}

#endif
