#include "welvet/vulkan/matvec.hpp"

#include "welvet/activate.hpp"

#include "vulkan/internal.hpp"
#include "vulkan/launch_utils.hpp"
#include "vulkan/pipelines.hpp"

#include <cstring>
#include <new>

namespace welvet::vulkan {
namespace {

Buffer* as_buf(void* p) { return static_cast<Buffer*>(p); }

Status alloc_slot(Buffer** out, std::size_t bytes) {
  *out = new (std::nothrow) Buffer();
  if (!*out) return Status::Err(StatusCode::Internal, "alloc buffer slot");
  auto st = buffer_alloc(bytes, *out);
  if (!st) {
    delete *out;
    *out = nullptr;
  }
  return st;
}

void free_slot(void** slot) {
  if (!slot || !*slot) return;
  Buffer* b = as_buf(*slot);
  buffer_release(b);
  delete b;
  *slot = nullptr;
}

struct MatvecPush {
  int batch;
  int out_dim;
  int in_dim;
};

struct BiasActPush {
  int bias_len;
  int act;
  int batch;
  int out_dim;
  int has_bias;
};

struct ActDerivPush {
  int act;
  int n;
};

} // namespace

Status resident_mount(ResidentDense* r, const double* W_f64, const double* bias_f64, int bias_len,
                      int batch, int out, int in, Activation act) {
  if (!r || !W_f64) return Status::Err(StatusCode::InvalidArgument, "null mount");
  if (batch <= 0 || out <= 0 || in <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "bad dims");
  }
  auto st = detail::ensure_runtime();
  if (!st) return st;
  if (!detail::runtime().shader_float64) {
    return Status::Err(StatusCode::Unsupported, "shaderFloat64 required for resident Dense");
  }

  resident_release(r);
  r->batch = batch;
  r->out = out;
  r->in = in;
  r->bias_len = bias_len;
  r->act = act;

  const std::size_t w_bytes = static_cast<std::size_t>(out) * static_cast<std::size_t>(in) * sizeof(double);
  const std::size_t x_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(in) * sizeof(float);
  const std::size_t y_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(out) * sizeof(float);

  Buffer *W = nullptr, *bias = nullptr, *x = nullptr, *pre = nullptr, *post = nullptr;
  st = alloc_slot(&W, w_bytes);
  if (!st) {
    resident_release(r);
    return st;
  }
  st = buffer_upload(W, W_f64, w_bytes);
  if (!st) {
    free_slot(reinterpret_cast<void**>(&W));
    resident_release(r);
    return st;
  }
  r->W = W;

  if (bias_f64 && bias_len > 0) {
    st = alloc_slot(&bias, static_cast<std::size_t>(bias_len) * sizeof(double));
    if (!st) {
      resident_release(r);
      return st;
    }
    st = buffer_upload(bias, bias_f64, static_cast<std::size_t>(bias_len) * sizeof(double));
    if (!st) {
      resident_release(r);
      return st;
    }
    r->bias = bias;
  }

  st = alloc_slot(&x, x_bytes);
  if (!st) {
    resident_release(r);
    return st;
  }
  r->x = x;
  st = alloc_slot(&pre, y_bytes);
  if (!st) {
    resident_release(r);
    return st;
  }
  r->pre = pre;
  st = alloc_slot(&post, y_bytes);
  if (!st) {
    resident_release(r);
    return st;
  }
  r->post = post;
  return Status::Ok();
}

Status resident_execute(ResidentDense* r, const float* x_host, float* pre_host, float* post_host) {
  if (!r || !r->W || !x_host || !pre_host || !post_host) {
    return Status::Err(StatusCode::InvalidArgument, "null execute");
  }
  Buffer* W = as_buf(r->W);
  Buffer* x = as_buf(r->x);
  Buffer* pre = as_buf(r->pre);
  Buffer* post = as_buf(r->post);

  const std::size_t x_bytes =
      static_cast<std::size_t>(r->batch) * static_cast<std::size_t>(r->in) * sizeof(float);
  const std::size_t y_bytes =
      static_cast<std::size_t>(r->batch) * static_cast<std::size_t>(r->out) * sizeof(float);

  auto st = buffer_upload(x, x_host, x_bytes);
  if (!st) return st;

  MatvecPush mp{r->batch, r->out, r->in};
  VkDescriptorBufferInfo bufs[3] = {detail::buf_info(*W, W->bytes), detail::buf_info(*x, x_bytes),
                                    detail::buf_info(*pre, y_bytes)};
  const int n = r->batch * r->out;
  const std::uint32_t groups = static_cast<std::uint32_t>((static_cast<std::size_t>(n) + 255) / 256);
  st = detail::dispatch_compute(detail::PipelineId::DenseMatvec, bufs, 3, &mp, sizeof(mp), groups);
  if (!st) return st;

  const bool need_bias_act =
      r->act != Activation::Linear || (r->bias != nullptr && r->bias_len > 0);
  if (need_bias_act) {
    Buffer* bias = r->bias ? as_buf(r->bias) : nullptr;
    VkDescriptorBufferInfo bbufs[3] = {detail::buf_info(*pre, y_bytes), detail::buf_info(*post, y_bytes),
                                       bias ? detail::buf_info(*bias, bias->bytes)
                                            : detail::buf_info(*pre, sizeof(float))};
    BiasActPush bp{r->bias_len, static_cast<int>(r->act), r->batch, r->out, bias ? 1 : 0};
    st = detail::dispatch_compute(detail::PipelineId::DenseBiasAct, bbufs, 3, &bp, sizeof(bp), groups);
    if (!st) return st;
  } else {
    std::memcpy(post->mapped, pre->mapped, y_bytes);
  }

  st = buffer_download(*pre, pre_host, y_bytes);
  if (!st) return st;
  return buffer_download(*post, post_host, y_bytes);
}

Status resident_backward(ResidentDense* r, const float* grad_out, const float* input,
                         const float* pre, float* grad_in, float* grad_w, bool linear) {
  if (!r || !r->W || !grad_out || !input || !pre || !grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "null resident backward");
  }

  const std::size_t y_bytes =
      static_cast<std::size_t>(r->batch) * static_cast<std::size_t>(r->out) * sizeof(float);
  const std::size_t x_bytes =
      static_cast<std::size_t>(r->batch) * static_cast<std::size_t>(r->in) * sizeof(float);
  const std::size_t dw_bytes =
      static_cast<std::size_t>(r->out) * static_cast<std::size_t>(r->in) * sizeof(float);

  Buffer gy{}, pre_dev{}, dpre_out{}, xb{}, gx{}, dw{};
  auto cleanup = [&]() {
    buffer_release(&gy);
    buffer_release(&pre_dev);
    buffer_release(&dpre_out);
    buffer_release(&xb);
    buffer_release(&gx);
    buffer_release(&dw);
  };

  auto st = buffer_alloc(y_bytes, &gy);
  if (!st) return st;
  st = buffer_alloc(y_bytes, &pre_dev);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(y_bytes, &dpre_out);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_alloc(x_bytes, &xb);
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

  st = buffer_upload(&gy, grad_out, y_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&pre_dev, pre, y_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_upload(&xb, input, x_bytes);
  if (!st) {
    cleanup();
    return st;
  }

  const int act_i = static_cast<int>(linear ? Activation::Linear : r->act);
  const int n_y = r->batch * r->out;
  const int n_x = r->batch * r->in;
  const int n_w = r->out * r->in;

  ActDerivPush ap{act_i, n_y};
  VkDescriptorBufferInfo ad_bufs[3] = {detail::buf_info(gy, y_bytes), detail::buf_info(pre_dev, y_bytes),
                                       detail::buf_info(dpre_out, y_bytes)};
  st = detail::dispatch_compute(
      detail::PipelineId::DenseActDeriv, ad_bufs, 3, &ap, sizeof(ap),
      static_cast<std::uint32_t>((static_cast<std::size_t>(n_y) + 255) / 256));
  if (!st) {
    cleanup();
    return st;
  }

  MatvecPush mp{r->batch, r->out, r->in};
  Buffer* W = as_buf(r->W);
  VkDescriptorBufferInfo mt_bufs[3] = {detail::buf_info(*W, W->bytes),
                                       detail::buf_info(dpre_out, y_bytes),
                                       detail::buf_info(gx, x_bytes)};
  st = detail::dispatch_compute(
      detail::PipelineId::DenseMatvect, mt_bufs, 3, &mp, sizeof(mp),
      static_cast<std::uint32_t>((static_cast<std::size_t>(n_x) + 255) / 256));
  if (!st) {
    cleanup();
    return st;
  }

  VkDescriptorBufferInfo ow_bufs[3] = {detail::buf_info(dpre_out, y_bytes), detail::buf_info(xb, x_bytes),
                                         detail::buf_info(dw, dw_bytes)};
  st = detail::dispatch_compute(
      detail::PipelineId::DenseOuter, ow_bufs, 3, &mp, sizeof(mp),
      static_cast<std::uint32_t>((static_cast<std::size_t>(n_w) + 255) / 256));
  if (!st) {
    cleanup();
    return st;
  }

  st = buffer_download(gx, grad_in, x_bytes);
  if (!st) {
    cleanup();
    return st;
  }
  st = buffer_download(dw, grad_w, dw_bytes);
  cleanup();
  return st;
}

void resident_release(ResidentDense* r) {
  if (!r) return;
  free_slot(reinterpret_cast<void**>(&r->W));
  free_slot(reinterpret_cast<void**>(&r->bias));
  free_slot(reinterpret_cast<void**>(&r->x));
  free_slot(reinterpret_cast<void**>(&r->pre));
  free_slot(reinterpret_cast<void**>(&r->post));
  *r = {};
}

} // namespace welvet::vulkan
