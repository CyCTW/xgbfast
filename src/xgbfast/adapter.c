/* Compile against the actual generated header; expose our own versioned ABI. */
#include "header.h"

typedef struct { union Entry* entries; size_t features; } XFContext;

uint32_t xf_abi_version(void) { return 1; }
int32_t xf_num_features(void) { return get_num_feature(); }

void* xf_create(void) {
  if (N_TARGET * MAX_N_CLASS != 1 || get_num_feature() < 1 ||
      strcmp(get_threshold_type(), "float32") || strcmp(get_leaf_output_type(), "float32")) return NULL;
  XFContext* ctx = (XFContext*)malloc(sizeof(XFContext));
  if (!ctx) return NULL;
  ctx->features = (size_t)get_num_feature();
  ctx->entries = (union Entry*)malloc(ctx->features * sizeof(union Entry));
  if (!ctx->entries) { free(ctx); return NULL; }
  return ctx;
}

void xf_free(void* handle) {
  XFContext* ctx = (XFContext*)handle;
  if (ctx) { free(ctx->entries); free(ctx); }
}

/* 0 = success, 1 = bad pointers/shape/capacity, 2 = infinite input.
 * Caller must serialize access to each context and keep input/output alive. */
int xf_predict(void* handle, const float* input, size_t rows, size_t features,
               float* output, size_t output_count) {
  XFContext* ctx = (XFContext*)handle;
  if (!ctx || features != ctx->features || output_count < rows ||
      (rows && (!input || !output))) return 1;
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < features; ++col) {
      float value = input[row * features + col];
      if (isnan(value)) ctx->entries[col].missing = -1;
      else if (isinf(value)) return 2;
      else ctx->entries[col].fvalue = value;
    }
    output[row] = 0.0f;
    predict(ctx->entries, 0, &output[row]);
  }
  return 0;
}
