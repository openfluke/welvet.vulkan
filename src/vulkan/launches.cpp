#include "welvet/vulkan/cameral.hpp"
#include "welvet/vulkan/conv.hpp"
#include "welvet/vulkan/embedding.hpp"
#include "welvet/vulkan/mha.hpp"
#include "welvet/vulkan/norm.hpp"
#include "welvet/vulkan/softmax.hpp"
#include "welvet/vulkan/swiglu.hpp"
#include "welvet/activate.hpp"
#include "welvet/vulkan/tween.hpp"

#include "vulkan/internal.hpp"
#include "vulkan/launch_utils.hpp"
#include "vulkan/pipelines.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
namespace welvet::vulkan {
namespace {

std::uint32_t groups256(std::size_t n) { return static_cast<std::uint32_t>((n + 255) / 256); }

struct NormFwdPush {
  int32_t n_tok;
  int32_t dim;
  double eps;
  int32_t center;
  int32_t has_beta;
};

struct NormBwdPush {
  int32_t n_tok;
  int32_t dim;
  double eps;
  int32_t center;
};

struct NormParamPush {
  int32_t n_tok;
  int32_t dim;
  int32_t center;
};

// Host MHA helpers — mirror welvet.cuda mha_* / welvet.cpp attn_* with libm f64
// (Vulkan poly sin/cos/pow miss Go exact_bit on FormatNone×34).
bool mha_attn_allow(int q_pos, int k_pos, int mask, int window, int msl, int prefix_len,
                    int window_causal) {
  if (mask == 2) return true; // Bidirectional
  if (mask == 3) {            // SlidingWindow
    const int w = window > 0 ? window : msl;
    if (window_causal) return k_pos <= q_pos && k_pos >= q_pos - w + 1;
    int d = q_pos - k_pos;
    if (d < 0) d = -d;
    return d < w;
  }
  if (mask == 4) { // PrefixLM
    if (q_pos < prefix_len && k_pos < prefix_len) return true;
    return k_pos <= q_pos;
  }
  return k_pos <= q_pos; // Causal / default
}

void mha_apply_rope(double* vec, int pos, int num_heads, int head_dim, double theta) {
  const int half = head_dim / 2;
  if (half <= 0 || theta <= 0.0) return;
  for (int h = 0; h < num_heads; ++h) {
    const int base = h * head_dim;
    for (int d = 0; d < half; ++d) {
      const double angle =
          static_cast<double>(pos) /
          std::pow(theta, static_cast<double>(2 * d) / static_cast<double>(head_dim));
      const double c = std::cos(angle);
      const double s = std::sin(angle);
      const double v0 = vec[base + d];
      const double v1 = vec[base + d + half];
      vec[base + d] = v0 * c - v1 * s;
      vec[base + d + half] = v0 * s + v1 * c;
    }
  }
}

void mha_apply_rope_backward(double* grad, int pos, int num_heads, int head_dim, double theta) {
  const int half = head_dim / 2;
  if (half <= 0 || theta <= 0.0) return;
  for (int h = 0; h < num_heads; ++h) {
    const int base = h * head_dim;
    for (int d = 0; d < half; ++d) {
      const double angle =
          static_cast<double>(pos) /
          std::pow(theta, static_cast<double>(2 * d) / static_cast<double>(head_dim));
      const double c = std::cos(angle);
      const double s = std::sin(angle);
      const double v0 = grad[base + d];
      const double v1 = grad[base + d + half];
      grad[base + d] = v0 * c + v1 * s;
      grad[base + d + half] = -v0 * s + v1 * c;
    }
  }
}

double mha_alibi_bias(int head, int num_heads, int q_pos, int k_pos, int use_alibi) {
  if (!use_alibi) return 0.0;
  int n = num_heads;
  if (n <= 0) n = 1;
  const double slope = std::pow(2.0, -static_cast<double>(head + 1) * 8.0 / static_cast<double>(n));
  const int diff = q_pos - k_pos;
  return -slope * static_cast<double>(diff < 0 ? -diff : diff);
}

} // namespace

Status launch_norm_forward(const float* x_host, const double* gamma_f64, const double* beta_f64,
                           float* pre_host, float* post_host, int n_tok, int dim, double eps,
                           int center) {
  if (!x_host || !gamma_f64 || !pre_host || !post_host) {
    return Status::Err(StatusCode::InvalidArgument, "norm: null ptr");
  }
  if (n_tok <= 0 || dim <= 0) return Status::Err(StatusCode::InvalidArgument, "norm: bad dims");
  if (center && !beta_f64) return Status::Err(StatusCode::InvalidArgument, "norm: LayerNorm needs beta");

  const std::size_t x_bytes = static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(dim) * sizeof(float);
  const std::size_t g_bytes = static_cast<std::size_t>(dim) * sizeof(double);

  Buffer x{}, pre{}, post{}, g{}, b{};
  auto cleanup = [&]() {
    buffer_release(&x);
    buffer_release(&pre);
    buffer_release(&post);
    buffer_release(&g);
    buffer_release(&b);
  };
  auto st = buffer_alloc(x_bytes, &x);
  if (!st) return st;
  st = buffer_alloc(x_bytes, &pre);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(x_bytes, &post);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(g_bytes, &g);
  if (!st) {
    cleanup();
    return st;
  }
  if (center) {
    st = buffer_alloc(g_bytes, &b);
    if (!st) {
      cleanup();
      return st;
    }
    st = buffer_upload(&b, beta_f64, g_bytes);
    if (!st) {
      cleanup();
      return st;
    }
  }
  st = buffer_upload(&x, x_host, x_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&g, gamma_f64, g_bytes);
  if (!st) {
    cleanup();
    return st;
  }

  NormFwdPush push{n_tok, dim, eps, center, center && beta_f64 ? 1 : 0};
  VkDescriptorBufferInfo bufs[5] = {
      detail::buf_info(x, x_bytes), detail::buf_info(g, g_bytes),
      center ? detail::buf_info(b, g_bytes) : detail::buf_info(g, sizeof(double)),
      detail::buf_info(pre, x_bytes), detail::buf_info(post, x_bytes)};
  st = detail::dispatch_compute(detail::PipelineId::NormForward, bufs, 5, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(n_tok)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(pre, pre_host, x_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(post, post_host, x_bytes);
  cleanup();
  return st;
}

Status launch_norm_backward(const float* x_host, const double* gamma_f64, const float* pre_host,
                            const float* grad_out_host, float* grad_in_host, float* dW_host,
                            int n_tok, int dim, double eps, int center) {
  if (!x_host || !gamma_f64 || !pre_host || !grad_out_host || !grad_in_host || !dW_host) {
    return Status::Err(StatusCode::InvalidArgument, "norm bwd: null ptr");
  }
  if (n_tok <= 0 || dim <= 0) return Status::Err(StatusCode::InvalidArgument, "norm bwd: bad dims");

  const std::size_t x_bytes = static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(dim) * sizeof(float);
  const std::size_t g_bytes = static_cast<std::size_t>(dim) * sizeof(double);
  const int dW_n = center ? 2 * dim : dim;
  const std::size_t dw_bytes = static_cast<std::size_t>(dW_n) * sizeof(float);

  Buffer x{}, pre{}, gy{}, gx{}, dw{}, g{};
  auto cleanup = [&]() {
    buffer_release(&x);
    buffer_release(&pre);
    buffer_release(&gy);
    buffer_release(&gx);
    buffer_release(&dw);
    buffer_release(&g);
  };
  auto st = buffer_alloc(x_bytes, &x);
  if (!st) return st;
  st = buffer_alloc(x_bytes, &pre);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(x_bytes, &gy);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(x_bytes, &gx);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(dw_bytes, &dw);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(g_bytes, &g);
  if (!st) {
    cleanup();
    return st;
  }
  std::memset(dw.mapped, 0, dw_bytes);
  st = buffer_upload(&x, x_host, x_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&pre, pre_host, x_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&gy, grad_out_host, x_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&g, gamma_f64, g_bytes);
  if (!st) {
    cleanup();
    return st;
  }

  NormBwdPush bpush{n_tok, dim, eps, center};
  VkDescriptorBufferInfo bbufs[5] = {detail::buf_info(x, x_bytes), detail::buf_info(g, g_bytes),
                                       detail::buf_info(pre, x_bytes), detail::buf_info(gy, x_bytes),
                                       detail::buf_info(gx, x_bytes)};
  st = detail::dispatch_compute(detail::PipelineId::NormBackward, bbufs, 5, &bpush, sizeof(bpush),
                                groups256(static_cast<std::size_t>(n_tok)));
  if (!st) {
    cleanup();
    return st;
  }

  NormParamPush ppush{n_tok, dim, center};
  VkDescriptorBufferInfo pbufs[3] = {detail::buf_info(pre, x_bytes), detail::buf_info(gy, x_bytes),
                                     detail::buf_info(dw, dw_bytes)};
  st = detail::dispatch_compute(detail::PipelineId::NormParamGrad, pbufs, 3, &ppush, sizeof(ppush),
                                groups256(static_cast<std::size_t>(dim)));
  if (!st) {
    cleanup();
    return st;
  }

  st = buffer_download(gx, grad_in_host, x_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(dw, dW_host, dw_bytes);
  cleanup();
  return st;
}

Status launch_softmax_forward(const float* x_host, float* out_host, int rows, int cols, int n,
                              int kind, double temperature, double entmax_alpha,
                              std::uint32_t gumbel_seed, const std::int32_t* mask_host,
                              int mask_len) {
  if (!x_host || !out_host) return Status::Err(StatusCode::InvalidArgument, "softmax: null ptr");
  if (rows <= 0 || cols <= 0 || n <= 0 || rows * cols > n) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: bad dims");
  }
  // Host libm f64 exp/log — GPU poly welvet_d_exp misses Go exact_bit (~65–85%).
  // Mirrors welvet.cuda softmax_forward_row / welvet.cpp stable_softmax.
  const double inv_t = 1.0 / (temperature == 0.0 ? 1.0 : temperature);
  std::vector<float> logits(static_cast<std::size_t>(cols));
  std::vector<float> sorted(static_cast<std::size_t>(cols));

  for (int tid = 0; tid < rows; ++tid) {
    const int start = tid * cols;
    std::uint32_t state = gumbel_seed;
    if (!state) state = 0xC0FFEEu;
    if (kind == 2) {
      for (int i = 0; i < start; ++i) {
        state = state * 1664525u + 1013904223u;
      }
    }

    float mx = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < cols; ++i) {
      float v = static_cast<float>(static_cast<double>(x_host[start + i]) * inv_t);
      if (kind == 3 && start + i < mask_len && mask_host && !mask_host[start + i]) {
        v = -1e9f;
      }
      if (kind == 2) {
        state = state * 1664525u + 1013904223u;
        float u = static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
        if (u < 1e-10f) u = 1e-10f;
        v += -static_cast<float>(std::log(-std::log(static_cast<double>(u))));
      }
      logits[static_cast<std::size_t>(i)] = v;
      sorted[static_cast<std::size_t>(i)] = v;
      if (v > mx) mx = v;
    }

    float se = 0.0f;
    for (int i = 0; i < cols; ++i) {
      // Go: exp in float64, then round each term to Float32.
      out_host[start + i] =
          static_cast<float>(std::exp(static_cast<double>(logits[static_cast<std::size_t>(i)] - mx)));
      se += out_host[start + i];
    }
    for (int i = 0; i < cols; ++i) out_host[start + i] /= se;

    float alpha = static_cast<float>(entmax_alpha);
    if (alpha == 0.0f) alpha = 1.5f;
    if (kind == 4 || (kind == 5 && alpha > 1.0f)) {
      for (int i = 0; i < cols; ++i) {
        for (int k = i + 1; k < cols; ++k) {
          if (sorted[static_cast<std::size_t>(i)] < sorted[static_cast<std::size_t>(k)]) {
            const float tmp = sorted[static_cast<std::size_t>(i)];
            sorted[static_cast<std::size_t>(i)] = sorted[static_cast<std::size_t>(k)];
            sorted[static_cast<std::size_t>(k)] = tmp;
          }
        }
      }
      float cum = 0.0f;
      int count = 0;
      for (int i = 0; i < cols; ++i) {
        cum += sorted[static_cast<std::size_t>(i)];
        if (sorted[static_cast<std::size_t>(i)] - (cum - 1.0f) / static_cast<float>(i + 1) > 0.0f) {
          count = i + 1;
        } else {
          break;
        }
      }
      float tau = 0.0f;
      if (count) {
        cum = 0.0f;
        for (int i = 0; i < count; ++i) cum += sorted[static_cast<std::size_t>(i)];
        tau = (cum - 1.0f) / static_cast<float>(count);
      }
      float sum = 0.0f;
      for (int i = 0; i < cols; ++i) {
        const float sp = std::fmax(0.0f, logits[static_cast<std::size_t>(i)] - tau);
        if (kind == 4 || alpha >= 2.0f) {
          out_host[start + i] = sp;
        } else {
          out_host[start + i] = (2.0f - alpha) * out_host[start + i] + (alpha - 1.0f) * sp;
        }
        sum += out_host[start + i];
      }
      if (kind == 5 && alpha < 2.0f) {
        for (int i = 0; i < cols; ++i) out_host[start + i] /= sum;
      }
    }
  }
  return Status::Ok();
}

Status launch_softmax_backward(const float* pre_host, const float* grad_out_host, float* grad_in_host,
                               int rows, int cols, int n, int kind, double temperature,
                               const std::int32_t* mask_host, int mask_len) {
  if (!pre_host || !grad_out_host || !grad_in_host) {
    return Status::Err(StatusCode::InvalidArgument, "softmax bwd: null ptr");
  }
  if (rows <= 0 || cols <= 0 || n <= 0 || rows * cols > n) {
    return Status::Err(StatusCode::InvalidArgument, "softmax bwd: bad dims");
  }
  const double inv_t = 1.0 / (temperature == 0.0 ? 1.0 : temperature);
  for (int tid = 0; tid < rows; ++tid) {
    const int start = tid * cols;
    float dot = 0.0f;
    for (int i = 0; i < cols; ++i) {
      float g = grad_out_host[start + i];
      if (kind == 3 && start + i < mask_len && mask_host && !mask_host[start + i]) g = 0.0f;
      dot += g * pre_host[start + i];
    }
    for (int j = 0; j < cols; ++j) {
      float g = grad_out_host[start + j];
      if (kind == 3 && start + j < mask_len && mask_host && !mask_host[start + j]) g = 0.0f;
      const float gl = pre_host[start + j] * (g - dot);
      grad_in_host[start + j] = static_cast<float>(static_cast<double>(gl) * inv_t);
    }
  }
  return Status::Ok();
}

Status launch_swiglu_product(const float* gate_pre_host, const float* up_post_host, float* h_host,
                           int n) {
  if (!gate_pre_host || !up_post_host || !h_host || n <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: null/bad args");
  }
  // Gate/Up Dense stay on Vulkan; SiLU⊙ uses host libm like LSTM host tanh —
  // glslang has no double exp, and GPU poly/soft-double miss Go exact_bit.
  for (int i = 0; i < n; ++i) {
    const double silu = activate_f64(static_cast<double>(gate_pre_host[i]), Activation::Silu);
    const float g_f = static_cast<float>(silu);
    h_host[i] = static_cast<float>(static_cast<double>(g_f) * static_cast<double>(up_post_host[i]));
  }
  return Status::Ok();
}

Status launch_swiglu_bwd_product(const float* gate_pre_host, const float* up_post_host,
                                 const float* grad_h_host, float* d_up_host, float* d_gate_host,
                                 int n) {
  if (!gate_pre_host || !up_post_host || !grad_h_host || !d_up_host || !d_gate_host || n <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu bwd: null/bad args");
  }
  for (int i = 0; i < n; ++i) {
    const double gp = static_cast<double>(gate_pre_host[i]);
    const float silu_f = static_cast<float>(activate_f64(gp, Activation::Silu));
    const float dsilu_f = static_cast<float>(activate_deriv_f64(gp, Activation::Silu));
    const double dh = static_cast<double>(grad_h_host[i]);
    const double up = static_cast<double>(up_post_host[i]);
    d_up_host[i] = static_cast<float>(dh * static_cast<double>(silu_f));
    d_gate_host[i] = static_cast<float>(dh * up * static_cast<double>(dsilu_f));
  }
  return Status::Ok();
}

Status launch_embedding_gather(const float* tokens_host, const float* W_f32, float* out_host,
                               int n_tok, int emb, int vocab, int in_last, Activation act) {
  if (!tokens_host || !W_f32 || !out_host) {
    return Status::Err(StatusCode::InvalidArgument, "embedding gather: null");
  }
  const std::size_t tok_bytes = static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(in_last) * sizeof(float);
  const std::size_t w_bytes = static_cast<std::size_t>(vocab) * static_cast<std::size_t>(emb) * sizeof(float);
  const std::size_t out_bytes = static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(emb) * sizeof(float);
  Buffer tok{}, w{}, out{};
  auto cleanup = [&]() {
    buffer_release(&tok);
    buffer_release(&w);
    buffer_release(&out);
  };
  auto st = buffer_alloc(tok_bytes, &tok);
  if (!st) return st;
  st = buffer_alloc(w_bytes, &w);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(out_bytes, &out);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&tok, tokens_host, tok_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&w, W_f32, w_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  struct Push {
    int n_tok;
    int emb;
    int vocab;
    int in_last;
    int act;
  } push{n_tok, emb, vocab, in_last, static_cast<int>(act)};
  VkDescriptorBufferInfo bufs[3] = {detail::buf_info(tok, tok_bytes), detail::buf_info(w, w_bytes),
                                    detail::buf_info(out, out_bytes)};
  st = detail::dispatch_compute(detail::PipelineId::EmbeddingGather, bufs, 3, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(emb)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(out, out_host, out_bytes);
  cleanup();
  return st;
}

Status launch_embedding_scatter(const float* tokens_host, const float* grad_out_host,
                                float* dW_host, int n_tok, int emb, int vocab, int in_last) {
  if (!tokens_host || !grad_out_host || !dW_host) {
    return Status::Err(StatusCode::InvalidArgument, "embedding scatter: null");
  }
  const std::size_t tok_bytes = static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(in_last) * sizeof(float);
  const std::size_t go_bytes = static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(emb) * sizeof(float);
  const std::size_t dw_bytes = static_cast<std::size_t>(vocab) * static_cast<std::size_t>(emb) * sizeof(float);
  Buffer tok{}, go{}, dw{};
  auto cleanup = [&]() {
    buffer_release(&tok);
    buffer_release(&go);
    buffer_release(&dw);
  };
  auto st = buffer_alloc(tok_bytes, &tok);
  if (!st) return st;
  st = buffer_alloc(go_bytes, &go);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(dw_bytes, &dw);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&tok, tokens_host, tok_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&go, grad_out_host, go_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  struct Push {
    int n_tok;
    int emb;
    int vocab;
    int in_last;
  } push{n_tok, emb, vocab, in_last};
  VkDescriptorBufferInfo bufs[3] = {detail::buf_info(tok, tok_bytes), detail::buf_info(go, go_bytes),
                                    detail::buf_info(dw, dw_bytes)};
  st = detail::dispatch_compute(detail::PipelineId::EmbeddingScatter, bufs, 3, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(vocab) * static_cast<std::size_t>(emb)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(dw, dW_host, dw_bytes);
  cleanup();
  return st;
}

Status launch_mha_attention_forward(const float* q_host, const float* k_host, const float* v_host,
                                    float* attn_host, int batch, int seq, int num_heads,
                                    int num_kv_heads, int head_dim, int max_seq_len, int mask,
                                    int pos, int softmax, int window, int prefix_len,
                                    int window_causal, double rope_theta, double scale_override) {
  if (!q_host || !k_host || !v_host || !attn_host) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn: null ptr");
  }
  if (batch <= 0 || seq <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      max_seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn: bad dims");
  }
  if (num_heads % num_kv_heads != 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn: heads%kv!=0");
  }

  // Q/K/V/O Dense stay on Vulkan; attention+RoPE uses host libm like SwiGLU host SiLU —
  // GPU poly sin/cos/pow miss Go exact_bit on FormatNone×34.
  const int q_dim = num_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;
  const int heads_per_kv = num_heads / num_kv_heads;
  const int msl = max_seq_len;
  const int kv_len_max = batch * seq;
  double scale = scale_override;
  if (scale <= 0.0) scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const bool use_rope = (pos == 0 || pos == 3);
  const bool use_alibi = (pos == 2 || pos == 3);
  const bool use_sigmoid = (softmax == 1);

  std::vector<double> cache_k(static_cast<std::size_t>(msl) * static_cast<std::size_t>(kv_dim), 0.0);
  std::vector<double> cache_v(static_cast<std::size_t>(msl) * static_cast<std::size_t>(kv_dim), 0.0);
  std::vector<double> Q(static_cast<std::size_t>(seq) * static_cast<std::size_t>(q_dim));
  std::vector<double> scores(static_cast<std::size_t>(kv_len_max));
  std::vector<int> k_positions(static_cast<std::size_t>(kv_len_max));

  for (int b = 0; b < batch; ++b) {
    const int seq_base = b * seq;
    for (int s = 0; s < seq; ++s) {
      const int tok_pos = seq_base + s;
      const int tok = b * seq + s;
      for (int i = 0; i < q_dim; ++i) {
        Q[static_cast<std::size_t>(s * q_dim + i)] =
            static_cast<double>(q_host[tok * q_dim + i]);
      }
      double* k_row = cache_k.data() + (tok_pos % msl) * kv_dim;
      double* v_row = cache_v.data() + (tok_pos % msl) * kv_dim;
      for (int i = 0; i < kv_dim; ++i) {
        k_row[i] = static_cast<double>(k_host[tok * kv_dim + i]);
        v_row[i] = static_cast<double>(v_host[tok * kv_dim + i]);
      }
      if (use_rope) {
        mha_apply_rope(Q.data() + s * q_dim, tok_pos, num_heads, head_dim, rope_theta);
        mha_apply_rope(k_row, tok_pos, num_kv_heads, head_dim, rope_theta);
      }
    }

    const int kv_len = seq_base + seq;
    for (int s = 0; s < seq; ++s) {
      const int q_pos = seq_base + s;
      for (int h = 0; h < num_heads; ++h) {
        const int kv_head = h / heads_per_kv;
        const int q_off = s * q_dim + h * head_dim;
        const int a_off = (b * seq + s) * q_dim + h * head_dim;
        int n = 0;
        double max_score = -1e9;
        for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
          if (!mha_attn_allow(q_pos, k_pos, mask, window, msl, prefix_len, window_causal)) continue;
          const int k_idx = k_pos % msl;
          double dot = 0.0;
          const int k_base = k_idx * kv_dim + kv_head * head_dim;
          for (int d = 0; d < head_dim; ++d) {
            dot += Q[static_cast<std::size_t>(q_off + d)] * cache_k[static_cast<std::size_t>(k_base + d)];
          }
          const double score =
              dot * scale + mha_alibi_bias(h, num_heads, q_pos, k_pos, use_alibi ? 1 : 0);
          scores[static_cast<std::size_t>(n)] = score;
          k_positions[static_cast<std::size_t>(n)] = k_pos;
          if (score > max_score) max_score = score;
          ++n;
        }
        if (n == 0) {
          for (int d = 0; d < head_dim; ++d) attn_host[a_off + d] = 0.0f;
          continue;
        }
        if (use_sigmoid) {
          for (int i = 0; i < n; ++i) {
            scores[static_cast<std::size_t>(i)] =
                1.0 / (1.0 + std::exp(-scores[static_cast<std::size_t>(i)]));
          }
        } else {
          double exp_sum = 0.0;
          for (int i = 0; i < n; ++i) {
            scores[static_cast<std::size_t>(i)] =
                std::exp(scores[static_cast<std::size_t>(i)] - max_score);
            exp_sum += scores[static_cast<std::size_t>(i)];
          }
          const double inv = 1.0 / exp_sum;
          for (int i = 0; i < n; ++i) scores[static_cast<std::size_t>(i)] *= inv;
        }
        for (int d = 0; d < head_dim; ++d) {
          double sum = 0.0;
          for (int i = 0; i < n; ++i) {
            const int v_idx = k_positions[static_cast<std::size_t>(i)] % msl;
            sum += scores[static_cast<std::size_t>(i)] *
                   cache_v[static_cast<std::size_t>(v_idx * kv_dim + kv_head * head_dim + d)];
          }
          attn_host[a_off + d] = static_cast<float>(sum);
        }
      }
    }
  }
  return Status::Ok();
}

Status launch_mha_attention_backward(const float* q_host, const float* k_host, const float* v_host,
                                     const float* grad_attn_host, float* gQ_host, float* gK_host,
                                     float* gV_host, int batch, int seq, int num_heads,
                                     int num_kv_heads, int head_dim, int max_seq_len, int mask,
                                     int pos, int softmax, int window, int prefix_len,
                                     int window_causal, double rope_theta, double scale_override) {
  if (!q_host || !k_host || !v_host || !grad_attn_host || !gQ_host || !gK_host || !gV_host) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn bwd: null ptr");
  }
  if (batch <= 0 || seq <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      max_seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn bwd: bad dims");
  }
  if (num_heads % num_kv_heads != 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn bwd: heads%kv!=0");
  }

  const int q_dim = num_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;
  const int heads_per_kv = num_heads / num_kv_heads;
  const int msl = max_seq_len;
  const int bs = batch * seq;
  const int kv_len_max = batch * seq;
  double scale = scale_override;
  if (scale <= 0.0) scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  const bool use_rope = (pos == 0 || pos == 3);
  const bool use_alibi = (pos == 2 || pos == 3);
  const bool use_sigmoid = (softmax == 1);

  std::vector<double> cache_k(static_cast<std::size_t>(msl) * static_cast<std::size_t>(kv_dim));
  std::vector<double> cache_v(static_cast<std::size_t>(msl) * static_cast<std::size_t>(kv_dim));
  std::vector<double> Q(static_cast<std::size_t>(seq) * static_cast<std::size_t>(q_dim));
  std::vector<double> gQ(static_cast<std::size_t>(seq) * static_cast<std::size_t>(q_dim));
  std::vector<double> gK(static_cast<std::size_t>(msl) * static_cast<std::size_t>(kv_dim), 0.0);
  std::vector<double> gV(static_cast<std::size_t>(msl) * static_cast<std::size_t>(kv_dim), 0.0);
  std::vector<double> scores(static_cast<std::size_t>(kv_len_max));
  std::vector<double> dW(static_cast<std::size_t>(kv_len_max));
  std::vector<double> d_score(static_cast<std::size_t>(kv_len_max));
  std::vector<int> k_positions(static_cast<std::size_t>(kv_len_max));

  for (int i = 0; i < bs * q_dim; ++i) gQ_host[i] = 0.0f;
  for (int i = 0; i < bs * kv_dim; ++i) {
    gK_host[i] = 0.0f;
    gV_host[i] = 0.0f;
  }

  for (int b = 0; b < batch; ++b) {
    const int seq_base = b * seq;
    std::fill(cache_k.begin(), cache_k.end(), 0.0);
    std::fill(cache_v.begin(), cache_v.end(), 0.0);
    std::fill(gQ.begin(), gQ.end(), 0.0);

    for (int s = 0; s < seq; ++s) {
      const int tok_pos = seq_base + s;
      const int tok = b * seq + s;
      for (int i = 0; i < q_dim; ++i) {
        Q[static_cast<std::size_t>(s * q_dim + i)] =
            static_cast<double>(q_host[tok * q_dim + i]);
      }
      double* k_row = cache_k.data() + (tok_pos % msl) * kv_dim;
      double* v_row = cache_v.data() + (tok_pos % msl) * kv_dim;
      for (int i = 0; i < kv_dim; ++i) {
        k_row[i] = static_cast<double>(k_host[tok * kv_dim + i]);
        v_row[i] = static_cast<double>(v_host[tok * kv_dim + i]);
      }
      if (use_rope) {
        mha_apply_rope(Q.data() + s * q_dim, tok_pos, num_heads, head_dim, rope_theta);
        mha_apply_rope(k_row, tok_pos, num_kv_heads, head_dim, rope_theta);
      }
    }

    const int kv_len = seq_base + seq;
    for (int h = 0; h < num_heads; ++h) {
      const int kv_head = h / heads_per_kv;
      for (int q_pos_rel = 0; q_pos_rel < seq; ++q_pos_rel) {
        const int q_pos = seq_base + q_pos_rel;
        const int q_off = q_pos_rel * q_dim + h * head_dim;
        const int g_off_tok = (b * seq + q_pos_rel) * q_dim + h * head_dim;
        int n = 0;
        double max_score = -1e9;
        for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
          if (!mha_attn_allow(q_pos, k_pos, mask, window, msl, prefix_len, window_causal)) continue;
          const int k_idx = k_pos % msl;
          double dot = 0.0;
          const int k_base = k_idx * kv_dim + kv_head * head_dim;
          for (int d = 0; d < head_dim; ++d) {
            dot += Q[static_cast<std::size_t>(q_off + d)] * cache_k[static_cast<std::size_t>(k_base + d)];
          }
          const double score =
              dot * scale + mha_alibi_bias(h, num_heads, q_pos, k_pos, use_alibi ? 1 : 0);
          scores[static_cast<std::size_t>(n)] = score;
          k_positions[static_cast<std::size_t>(n)] = k_pos;
          if (score > max_score) max_score = score;
          ++n;
        }
        if (n == 0) continue;

        if (use_sigmoid) {
          for (int i = 0; i < n; ++i) {
            scores[static_cast<std::size_t>(i)] =
                1.0 / (1.0 + std::exp(-scores[static_cast<std::size_t>(i)]));
          }
        } else {
          double exp_sum = 0.0;
          for (int i = 0; i < n; ++i) {
            scores[static_cast<std::size_t>(i)] =
                std::exp(scores[static_cast<std::size_t>(i)] - max_score);
            exp_sum += scores[static_cast<std::size_t>(i)];
          }
          const double inv = 1.0 / exp_sum;
          for (int i = 0; i < n; ++i) scores[static_cast<std::size_t>(i)] *= inv;
        }

        for (int i = 0; i < n; ++i) dW[static_cast<std::size_t>(i)] = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          const double dy = static_cast<double>(grad_attn_host[g_off_tok + d]);
          for (int i = 0; i < n; ++i) {
            const int v_idx = k_positions[static_cast<std::size_t>(i)] % msl;
            const double v =
                cache_v[static_cast<std::size_t>(v_idx * kv_dim + kv_head * head_dim + d)];
            gV[static_cast<std::size_t>(v_idx * kv_dim + kv_head * head_dim + d)] +=
                scores[static_cast<std::size_t>(i)] * dy;
            dW[static_cast<std::size_t>(i)] += v * dy;
          }
        }

        if (use_sigmoid) {
          for (int i = 0; i < n; ++i) {
            const double sig = scores[static_cast<std::size_t>(i)];
            d_score[static_cast<std::size_t>(i)] =
                dW[static_cast<std::size_t>(i)] * sig * (1.0 - sig) * scale;
          }
        } else {
          double sum = 0.0;
          for (int i = 0; i < n; ++i) {
            sum += dW[static_cast<std::size_t>(i)] * scores[static_cast<std::size_t>(i)];
          }
          for (int i = 0; i < n; ++i) {
            d_score[static_cast<std::size_t>(i)] =
                scores[static_cast<std::size_t>(i)] *
                (dW[static_cast<std::size_t>(i)] - sum) * scale;
          }
        }

        for (int i = 0; i < n; ++i) {
          const int k_idx = k_positions[static_cast<std::size_t>(i)] % msl;
          const int k_base = k_idx * kv_dim + kv_head * head_dim;
          const double ds = d_score[static_cast<std::size_t>(i)];
          for (int d = 0; d < head_dim; ++d) {
            gQ[static_cast<std::size_t>(q_off + d)] +=
                ds * cache_k[static_cast<std::size_t>(k_base + d)];
            gK[static_cast<std::size_t>(k_base + d)] +=
                ds * Q[static_cast<std::size_t>(q_off + d)];
          }
        }
      }
    }

    for (int s = 0; s < seq; ++s) {
      const int tok_pos = seq_base + s;
      if (use_rope) {
        mha_apply_rope_backward(gQ.data() + s * q_dim, tok_pos, num_heads, head_dim, rope_theta);
      }
      const int tok = b * seq + s;
      for (int i = 0; i < q_dim; ++i) {
        gQ_host[tok * q_dim + i] = static_cast<float>(gQ[static_cast<std::size_t>(s * q_dim + i)]);
      }
    }
    if (use_rope) {
      for (int s = 0; s < seq; ++s) {
        const int tok_pos = seq_base + s;
        const int k_idx = tok_pos % msl;
        mha_apply_rope_backward(gK.data() + k_idx * kv_dim, tok_pos, num_kv_heads, head_dim,
                                rope_theta);
      }
    }
  }

  for (int tok = 0; tok < bs; ++tok) {
    const int k_idx = tok % msl;
    for (int i = 0; i < kv_dim; ++i) {
      gK_host[tok * kv_dim + i] =
          static_cast<float>(gK[static_cast<std::size_t>(k_idx * kv_dim + i)]);
      gV_host[tok * kv_dim + i] =
          static_cast<float>(gV[static_cast<std::size_t>(k_idx * kv_dim + i)]);
    }
  }
  return Status::Ok();
}

static Status convt_launch(detail::PipelineId pid, const float* x_host, const float* W_f32,
                           float* pre_host, float* post_host, std::size_t x_bytes, std::size_t w_bytes,
                           std::size_t out_bytes, int n_out, const void* push, std::size_t push_size) {
  Buffer x{}, w{}, pre{}, post{};
  auto cleanup = [&]() {
    buffer_release(&x);
    buffer_release(&w);
    buffer_release(&pre);
    buffer_release(&post);
  };
  auto st = buffer_alloc(x_bytes, &x);
  if (!st) return st;
  st = buffer_alloc(w_bytes, &w);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(out_bytes, &pre);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(out_bytes, &post);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&x, x_host, x_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&w, W_f32, w_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  VkDescriptorBufferInfo bufs[4] = {detail::buf_info(x, x_bytes), detail::buf_info(w, w_bytes),
                                    detail::buf_info(pre, out_bytes), detail::buf_info(post, out_bytes)};
  st = detail::dispatch_compute(pid, bufs, 4, push, push_size, groups256(static_cast<std::size_t>(n_out)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(pre, pre_host, out_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(post, post_host, out_bytes);
  cleanup();
  return st;
}

Status launch_convt1_forward(const float* x_host, const float* W_f32, float* pre_host,
                             float* post_host, int batch, int in_c, int filters, int seq, int out_len,
                             int kernel, int stride, int padding, Activation act) {
  if (!x_host || !W_f32 || !pre_host || !post_host) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: null ptr");
  }
  const int n_out = batch * filters * out_len;
  struct Push {
    int batch, in_c, filters, seq, out_len, kernel, stride, padding, act;
  } push{batch, in_c, filters, seq, out_len, kernel, stride, padding, static_cast<int>(act)};
  return convt_launch(detail::PipelineId::Convt1, x_host, W_f32, pre_host, post_host,
                      static_cast<std::size_t>(batch * in_c * seq) * sizeof(float),
                      static_cast<std::size_t>(filters * in_c * kernel) * sizeof(float),
                      static_cast<std::size_t>(n_out) * sizeof(float), n_out, &push, sizeof(push));
}

Status launch_convt2_forward(const float* x_host, const float* W_f32, float* pre_host,
                             float* post_host, int batch, int in_c, int filters, int in_h, int in_w,
                             int out_h, int out_w, int kernel, int stride, int padding,
                             Activation act) {
  if (!x_host || !W_f32 || !pre_host || !post_host) {
    return Status::Err(StatusCode::InvalidArgument, "convt2: null ptr");
  }
  const int sp_in = in_h * in_w;
  const int sp_out = out_h * out_w;
  const int n_out = batch * filters * sp_out;
  struct Push {
    int batch, in_c, filters, in_h, in_w, out_h, out_w, kernel, stride, padding, act;
  } push{batch, in_c, filters, in_h, in_w, out_h, out_w, kernel, stride, padding,
         static_cast<int>(act)};
  return convt_launch(detail::PipelineId::Convt2, x_host, W_f32, pre_host, post_host,
                      static_cast<std::size_t>(batch * in_c * sp_in) * sizeof(float),
                      static_cast<std::size_t>(filters * in_c * kernel * kernel) * sizeof(float),
                      static_cast<std::size_t>(n_out) * sizeof(float), n_out, &push, sizeof(push));
}

Status launch_convt3_forward(const float* x_host, const float* W_f32, float* pre_host,
                             float* post_host, int batch, int in_c, int filters, int in_d, int in_h,
                             int in_w, int out_d, int out_h, int out_w, int kernel, int stride,
                             int padding, Activation act) {
  if (!x_host || !W_f32 || !pre_host || !post_host) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: null ptr");
  }
  const int sp_in = in_d * in_h * in_w;
  const int sp_out = out_d * out_h * out_w;
  const int k3 = kernel * kernel * kernel;
  const int n_out = batch * filters * sp_out;
  struct Push {
    int batch, in_c, filters, in_d, in_h, in_w, out_d, out_h, out_w, kernel, stride, padding, act;
  } push{batch, in_c, filters, in_d, in_h, in_w, out_d, out_h, out_w, kernel, stride, padding,
         static_cast<int>(act)};
  return convt_launch(detail::PipelineId::Convt3, x_host, W_f32, pre_host, post_host,
                      static_cast<std::size_t>(batch * in_c * sp_in) * sizeof(float),
                      static_cast<std::size_t>(filters * in_c * k3) * sizeof(float),
                      static_cast<std::size_t>(n_out) * sizeof(float), n_out, &push, sizeof(push));
}

Status launch_project_gap(const float* src_host, float* dst_host, std::int32_t sn, std::int32_t dn) {
  if (!src_host || !dst_host) return Status::Err(StatusCode::InvalidArgument, "project_gap: null");
  if (sn <= 0 || dn <= 0) return Status::Err(StatusCode::InvalidArgument, "project_gap: empty");
  if (sn == dn) {
    std::memcpy(dst_host, src_host, static_cast<std::size_t>(dn) * sizeof(float));
    return Status::Ok();
  }
  Buffer src{}, dst{};
  auto cleanup = [&]() {
    buffer_release(&src);
    buffer_release(&dst);
  };
  auto st = buffer_alloc(static_cast<std::size_t>(sn) * sizeof(float), &src);
  if (!st) return st;
  st = buffer_alloc(static_cast<std::size_t>(dn) * sizeof(float), &dst);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&src, src_host, static_cast<std::size_t>(sn) * sizeof(float));
  if (!st) {
    cleanup();
    return st;
  }
  struct Push {
    std::uint32_t sn;
    std::uint32_t dn;
  } push{static_cast<std::uint32_t>(sn), static_cast<std::uint32_t>(dn)};
  VkDescriptorBufferInfo bufs[2] = {detail::buf_info(src, src.bytes), detail::buf_info(dst, dst.bytes)};
  st = detail::dispatch_compute(detail::PipelineId::ProjectGap, bufs, 2, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(dn)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(dst, dst_host, static_cast<std::size_t>(dn) * sizeof(float));
  cleanup();
  return st;
}

Status launch_cam_shadow(float* gy, const float* student, const float* teacher, std::int32_t n,
                         double coef) {
  if (!gy || !student || !teacher || n <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cam_shadow args");
  }
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  Buffer d_gy{}, d_st{}, d_te{};
  auto cleanup = [&]() {
    buffer_release(&d_gy);
    buffer_release(&d_st);
    buffer_release(&d_te);
  };
  auto st = buffer_alloc(bytes, &d_gy);
  if (!st) return st;
  st = buffer_alloc(bytes, &d_st);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(bytes, &d_te);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&d_gy, gy, bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&d_st, student, bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&d_te, teacher, bytes);
  if (!st) {
    cleanup();
    return st;
  }
  struct Push {
    std::uint32_t n;
    double coef;
  } push{static_cast<std::uint32_t>(n), coef};
  VkDescriptorBufferInfo bufs[3] = {detail::buf_info(d_gy, bytes), detail::buf_info(d_st, bytes),
                                    detail::buf_info(d_te, bytes)};
  st = detail::dispatch_compute(detail::PipelineId::CamShadow, bufs, 3, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(n)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(d_gy, gy, bytes);
  cleanup();
  return st;
}

Status launch_cam_gate_gap(const float* gy, std::int32_t n, float* out, std::int32_t out_n) {
  if (!gy || !out || n <= 0 || out_n <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "cam_gate_gap args");
  }
  Buffer d_gy{}, d_out{};
  auto cleanup = [&]() {
    buffer_release(&d_gy);
    buffer_release(&d_out);
  };
  auto st = buffer_alloc(static_cast<std::size_t>(n) * sizeof(float), &d_gy);
  if (!st) return st;
  st = buffer_alloc(static_cast<std::size_t>(out_n) * sizeof(float), &d_out);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&d_gy, gy, static_cast<std::size_t>(n) * sizeof(float));
  if (!st) {
    cleanup();
    return st;
  }
  struct Push {
    std::uint32_t n;
    std::uint32_t out_n;
  } push{static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(out_n)};
  VkDescriptorBufferInfo bufs[2] = {detail::buf_info(d_gy, d_gy.bytes), detail::buf_info(d_out, d_out.bytes)};
  st = detail::dispatch_compute(detail::PipelineId::CamGateGap, bufs, 2, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(out_n)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(d_out, out, static_cast<std::size_t>(out_n) * sizeof(float));
  cleanup();
  return st;
}

Status launch_cam_blend(const float* packed_host, const double* alphas_host, float* blended_host,
                        std::int32_t n, std::int32_t nb) {
  if (!packed_host || !alphas_host || !blended_host || n <= 0 || nb < 2) {
    return Status::Err(StatusCode::InvalidArgument, "cam_blend args");
  }
  const std::size_t pbytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(nb) * sizeof(float);
  const std::size_t abytes = static_cast<std::size_t>(nb) * sizeof(double);
  Buffer pack{}, out{}, alphas{};
  auto cleanup = [&]() {
    buffer_release(&pack);
    buffer_release(&out);
    buffer_release(&alphas);
  };
  auto st = buffer_alloc(pbytes, &pack);
  if (!st) return st;
  st = buffer_alloc(pbytes, &out);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(abytes, &alphas);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&pack, packed_host, pbytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&alphas, alphas_host, abytes);
  if (!st) {
    cleanup();
    return st;
  }
  struct Push {
    std::uint32_t n;
    std::uint32_t nb;
  } push{static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(nb)};
  VkDescriptorBufferInfo bufs[3] = {detail::buf_info(pack, pbytes), detail::buf_info(alphas, abytes),
                                    detail::buf_info(out, pbytes)};
  st = detail::dispatch_compute(detail::PipelineId::CamBlend, bufs, 3, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(n)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(out, blended_host, pbytes);
  cleanup();
  return st;
}

Status launch_cam_bias_blend(const double* biases_host, const double* alphas_host,
                             double* blended_host, std::int32_t n, std::int32_t nb) {
  if (!biases_host || !alphas_host || !blended_host || n <= 0 || nb < 2) {
    return Status::Err(StatusCode::InvalidArgument, "cam_bias_blend args");
  }
  const std::size_t bbytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(nb) * sizeof(double);
  const std::size_t abytes = static_cast<std::size_t>(nb) * sizeof(double);
  Buffer bias{}, out{}, alphas{};
  auto cleanup = [&]() {
    buffer_release(&bias);
    buffer_release(&out);
    buffer_release(&alphas);
  };
  auto st = buffer_alloc(bbytes, &bias);
  if (!st) return st;
  st = buffer_alloc(bbytes, &out);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(abytes, &alphas);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&bias, biases_host, bbytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&alphas, alphas_host, abytes);
  if (!st) {
    cleanup();
    return st;
  }
  struct Push {
    std::uint32_t n;
    std::uint32_t nb;
  } push{static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(nb)};
  VkDescriptorBufferInfo bufs[3] = {detail::buf_info(bias, bbytes), detail::buf_info(alphas, abytes),
                                    detail::buf_info(out, bbytes)};
  st = detail::dispatch_compute(detail::PipelineId::CamBiasBlend, bufs, 3, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(n)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(out, blended_host, bbytes);
  cleanup();
  return st;
}

} // namespace welvet::vulkan
