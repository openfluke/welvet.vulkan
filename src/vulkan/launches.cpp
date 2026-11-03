#include "welvet/vulkan/cameral.hpp"
#include "welvet/vulkan/conv.hpp"
#include "welvet/vulkan/embedding.hpp"
#include "welvet/vulkan/mha.hpp"
#include "welvet/vulkan/norm.hpp"
#include "welvet/vulkan/softmax.hpp"
#include "welvet/vulkan/swiglu.hpp"
#include "welvet/vulkan/tween.hpp"

#include "vulkan/internal.hpp"
#include "vulkan/launch_utils.hpp"
#include "vulkan/pipelines.hpp"

#include <cstdint>
#include <cstring>
namespace welvet::vulkan {
namespace {

std::uint32_t groups256(std::size_t n) { return static_cast<std::uint32_t>((n + 255) / 256); }

struct SoftmaxFwdPush {
  int32_t rows;
  int32_t cols;
  int32_t kind;
  int32_t mask_len;
  std::uint32_t gumbel_seed;
  double temperature;
  float entmax_alpha;
  int32_t has_mask;
};

struct SoftmaxBwdPush {
  int32_t rows;
  int32_t cols;
  int32_t kind;
  int32_t mask_len;
  double temperature;
  int32_t has_mask;
};

struct MhaPush {
  int32_t batch;
  int32_t seq;
  int32_t num_heads;
  int32_t num_kv;
  int32_t head_dim;
  int32_t msl;
  int32_t mask;
  int32_t pos_kind;
  int32_t softmax_kind;
  int32_t window;
  int32_t prefix_len;
  int32_t window_causal;
  double rope_theta;
  double scale_override;
};

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

  const std::size_t x_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t work_bytes =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(float);
  const int has_mask = (mask_host != nullptr && mask_len > 0) ? 1 : 0;
  const std::size_t mask_bytes =
      has_mask ? static_cast<std::size_t>(mask_len) * sizeof(std::int32_t) : sizeof(std::int32_t);

  Buffer x{}, out{}, logits{}, sorted{}, mask{};
  auto cleanup = [&]() {
    buffer_release(&x);
    buffer_release(&out);
    buffer_release(&logits);
    buffer_release(&sorted);
    buffer_release(&mask);
  };

  auto st = buffer_alloc(x_bytes, &x);
  if (!st) return st;
  st = buffer_alloc(x_bytes, &out);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(work_bytes, &logits);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(work_bytes, &sorted);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(mask_bytes, &mask);
  if (!st) {
    cleanup();
    return st;
  }

  std::memset(out.mapped, 0, x_bytes);
  st = buffer_upload(&x, x_host, x_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  if (has_mask) {
    st = buffer_upload(&mask, mask_host, mask_bytes);
  } else {
    std::memset(mask.mapped, 0, mask_bytes);
  }
  if (!st) {
    cleanup();
    return st;
  }

  SoftmaxFwdPush push{rows,
                      cols,
                      kind,
                      mask_len,
                      gumbel_seed,
                      temperature,
                      static_cast<float>(entmax_alpha),
                      has_mask};
  VkDescriptorBufferInfo bufs[5] = {detail::buf_info(x, x_bytes), detail::buf_info(out, x_bytes),
                                    detail::buf_info(logits, work_bytes),
                                    detail::buf_info(sorted, work_bytes),
                                    detail::buf_info(mask, mask_bytes)};
  st = detail::dispatch_compute(detail::PipelineId::SoftmaxForward, bufs, 5, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(rows)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(out, out_host, x_bytes);
  cleanup();
  return st;
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

  const std::size_t x_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const int has_mask = (mask_host != nullptr && mask_len > 0) ? 1 : 0;
  const std::size_t mask_bytes =
      has_mask ? static_cast<std::size_t>(mask_len) * sizeof(std::int32_t) : sizeof(std::int32_t);

  Buffer pre{}, gy{}, gx{}, mask{};
  auto cleanup = [&]() {
    buffer_release(&pre);
    buffer_release(&gy);
    buffer_release(&gx);
    buffer_release(&mask);
  };

  auto st = buffer_alloc(x_bytes, &pre);
  if (!st) return st;
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
  st = buffer_alloc(mask_bytes, &mask);
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
  if (has_mask) {
    st = buffer_upload(&mask, mask_host, mask_bytes);
  } else {
    std::memset(mask.mapped, 0, mask_bytes);
  }
  if (!st) {
    cleanup();
    return st;
  }

  SoftmaxBwdPush push{rows, cols, kind, mask_len, temperature, has_mask};
  VkDescriptorBufferInfo bufs[4] = {detail::buf_info(pre, x_bytes), detail::buf_info(gy, x_bytes),
                                    detail::buf_info(gx, x_bytes), detail::buf_info(mask, mask_bytes)};
  st = detail::dispatch_compute(detail::PipelineId::SoftmaxBackward, bufs, 4, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(rows)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(gx, grad_in_host, x_bytes);
  cleanup();
  return st;
}

Status launch_swiglu_product(const float* gate_pre_host, const float* up_post_host, float* h_host,
                             int n) {
  if (!gate_pre_host || !up_post_host || !h_host || n <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: null/bad args");
  }

  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  Buffer gate{}, up{}, h{};
  auto cleanup = [&]() {
    buffer_release(&gate);
    buffer_release(&up);
    buffer_release(&h);
  };

  auto st = buffer_alloc(bytes, &gate);
  if (!st) return st;
  st = buffer_alloc(bytes, &up);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(bytes, &h);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&gate, gate_pre_host, bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&up, up_post_host, bytes);
  if (!st) {
    cleanup();
    return st;
  }

  int push = n;
  VkDescriptorBufferInfo bufs[3] = {detail::buf_info(gate, bytes), detail::buf_info(up, bytes),
                                    detail::buf_info(h, bytes)};
  st = detail::dispatch_compute(detail::PipelineId::SwigluProduct, bufs, 3, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(n)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(h, h_host, bytes);
  cleanup();
  return st;
}

Status launch_swiglu_bwd_product(const float* gate_pre_host, const float* up_post_host,
                                 const float* grad_h_host, float* d_up_host, float* d_gate_host,
                                 int n) {
  if (!gate_pre_host || !up_post_host || !grad_h_host || !d_up_host || !d_gate_host || n <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu bwd: null/bad args");
  }

  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  Buffer gate{}, up{}, gh{}, dup{}, dg{};
  auto cleanup = [&]() {
    buffer_release(&gate);
    buffer_release(&up);
    buffer_release(&gh);
    buffer_release(&dup);
    buffer_release(&dg);
  };

  auto st = buffer_alloc(bytes, &gate);
  if (!st) return st;
  st = buffer_alloc(bytes, &up);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(bytes, &gh);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(bytes, &dup);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(bytes, &dg);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&gate, gate_pre_host, bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&up, up_post_host, bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&gh, grad_h_host, bytes);
  if (!st) {
    cleanup();
    return st;
  }

  int push = n;
  VkDescriptorBufferInfo bufs[5] = {detail::buf_info(gate, bytes), detail::buf_info(up, bytes),
                                    detail::buf_info(gh, bytes), detail::buf_info(dup, bytes),
                                    detail::buf_info(dg, bytes)};
  st = detail::dispatch_compute(detail::PipelineId::SwigluBwd, bufs, 5, &push, sizeof(push),
                                groups256(static_cast<std::size_t>(n)));
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(dup, d_up_host, bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(dg, d_gate_host, bytes);
  cleanup();
  return st;
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

  const int q_dim = num_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;
  const int bs = batch * seq;
  const int kv_len_max = batch * seq;

  const std::size_t q_bytes =
      static_cast<std::size_t>(bs) * static_cast<std::size_t>(q_dim) * sizeof(float);
  const std::size_t k_bytes =
      static_cast<std::size_t>(bs) * static_cast<std::size_t>(kv_dim) * sizeof(float);
  const std::size_t cache_bytes =
      static_cast<std::size_t>(max_seq_len) * static_cast<std::size_t>(kv_dim) * sizeof(double);
  const std::size_t q_work_bytes =
      static_cast<std::size_t>(seq) * static_cast<std::size_t>(q_dim) * sizeof(double);
  const std::size_t scores_bytes = static_cast<std::size_t>(kv_len_max) * sizeof(double);
  const std::size_t kpos_bytes = static_cast<std::size_t>(kv_len_max) * sizeof(int32_t);

  Buffer q{}, k{}, v{}, attn{}, cache_k{}, cache_v{}, q_work{}, scores{}, kpos{};
  auto cleanup = [&]() {
    buffer_release(&q);
    buffer_release(&k);
    buffer_release(&v);
    buffer_release(&attn);
    buffer_release(&cache_k);
    buffer_release(&cache_v);
    buffer_release(&q_work);
    buffer_release(&scores);
    buffer_release(&kpos);
  };

  auto st = buffer_alloc(q_bytes, &q);
  if (!st) return st;
  st = buffer_alloc(k_bytes, &k);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(k_bytes, &v);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(q_bytes, &attn);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(cache_bytes, &cache_k);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(cache_bytes, &cache_v);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(q_work_bytes, &q_work);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(scores_bytes, &scores);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(kpos_bytes, &kpos);
  if (!st) {
    cleanup();
    return st;
  }

  st = buffer_upload(&q, q_host, q_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&k, k_host, k_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&v, v_host, k_bytes);
  if (!st) {
    cleanup();
    return st;
  }

  MhaPush push{batch,
               seq,
               num_heads,
               num_kv_heads,
               head_dim,
               max_seq_len,
               mask,
               pos,
               softmax,
               window,
               prefix_len,
               window_causal,
               rope_theta,
               scale_override};
  VkDescriptorBufferInfo bufs[9] = {
      detail::buf_info(q, q_bytes),         detail::buf_info(k, k_bytes),
      detail::buf_info(v, k_bytes),         detail::buf_info(attn, q_bytes),
      detail::buf_info(cache_k, cache_bytes), detail::buf_info(cache_v, cache_bytes),
      detail::buf_info(q_work, q_work_bytes), detail::buf_info(scores, scores_bytes),
      detail::buf_info(kpos, kpos_bytes)};
  st = detail::dispatch_compute(detail::PipelineId::MhaForward, bufs, 9, &push, sizeof(push), 1, 1, 1);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(attn, attn_host, q_bytes);
  cleanup();
  return st;
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
  const int bs = batch * seq;
  const int kv_len_max = batch * seq;

  const std::size_t q_bytes =
      static_cast<std::size_t>(bs) * static_cast<std::size_t>(q_dim) * sizeof(float);
  const std::size_t k_bytes =
      static_cast<std::size_t>(bs) * static_cast<std::size_t>(kv_dim) * sizeof(float);
  const std::size_t cache_bytes =
      static_cast<std::size_t>(max_seq_len) * static_cast<std::size_t>(kv_dim) * sizeof(double);
  const std::size_t q_work_bytes =
      static_cast<std::size_t>(seq) * static_cast<std::size_t>(q_dim) * sizeof(double);
  const std::size_t scores_bytes = static_cast<std::size_t>(kv_len_max) * sizeof(double);
  const std::size_t kpos_bytes = static_cast<std::size_t>(kv_len_max) * sizeof(int32_t);

  Buffer q{}, k{}, v{}, gy{}, gq_out{}, gk_out{}, gv_out{}, cache_k{}, cache_v{}, q_work{}, gq{},
      gk{}, gv{}, scores{}, dw{}, dscore{}, kpos{};
  auto cleanup = [&]() {
    buffer_release(&q);
    buffer_release(&k);
    buffer_release(&v);
    buffer_release(&gy);
    buffer_release(&gq_out);
    buffer_release(&gk_out);
    buffer_release(&gv_out);
    buffer_release(&cache_k);
    buffer_release(&cache_v);
    buffer_release(&q_work);
    buffer_release(&gq);
    buffer_release(&gk);
    buffer_release(&gv);
    buffer_release(&scores);
    buffer_release(&dw);
    buffer_release(&dscore);
    buffer_release(&kpos);
  };

  auto st = buffer_alloc(q_bytes, &q);
  if (!st) return st;
  st = buffer_alloc(k_bytes, &k);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(k_bytes, &v);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(q_bytes, &gy);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(q_bytes, &gq_out);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(k_bytes, &gk_out);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(k_bytes, &gv_out);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(cache_bytes, &cache_k);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(cache_bytes, &cache_v);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(q_work_bytes, &q_work);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(q_work_bytes, &gq);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(cache_bytes, &gk);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(cache_bytes, &gv);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(scores_bytes, &scores);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(scores_bytes, &dw);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(scores_bytes, &dscore);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(kpos_bytes, &kpos);
  if (!st) {
    cleanup();
    return st;
  }

  st = buffer_upload(&q, q_host, q_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&k, k_host, k_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&v, v_host, k_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&gy, grad_attn_host, q_bytes);
  if (!st) {
    cleanup();
    return st;
  }

  MhaPush push{batch,
               seq,
               num_heads,
               num_kv_heads,
               head_dim,
               max_seq_len,
               mask,
               pos,
               softmax,
               window,
               prefix_len,
               window_causal,
               rope_theta,
               scale_override};
  VkDescriptorBufferInfo bufs[17] = {
      detail::buf_info(q, q_bytes),
      detail::buf_info(k, k_bytes),
      detail::buf_info(v, k_bytes),
      detail::buf_info(gy, q_bytes),
      detail::buf_info(gq_out, q_bytes),
      detail::buf_info(gk_out, k_bytes),
      detail::buf_info(gv_out, k_bytes),
      detail::buf_info(cache_k, cache_bytes),
      detail::buf_info(cache_v, cache_bytes),
      detail::buf_info(q_work, q_work_bytes),
      detail::buf_info(gq, q_work_bytes),
      detail::buf_info(gk, cache_bytes),
      detail::buf_info(gv, cache_bytes),
      detail::buf_info(scores, scores_bytes),
      detail::buf_info(dw, scores_bytes),
      detail::buf_info(dscore, scores_bytes),
      detail::buf_info(kpos, kpos_bytes)};
  st = detail::dispatch_compute(detail::PipelineId::MhaBackward, bufs, 17, &push, sizeof(push), 1, 1,
                                1);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(gq_out, gQ_host, q_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(gk_out, gK_host, k_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(gv_out, gV_host, k_bytes);
  cleanup();
  return st;
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
