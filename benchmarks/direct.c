/* Compiled alongside TL2cgen's generated code, using its actual ABI. */
#include "header.h"

#ifdef _WIN32
#define DIRECT_EXPORT __declspec(dllexport)
#else
#define DIRECT_EXPORT
#endif

typedef struct {
  union Entry* entries;
  size_t num_feature;
} DirectContext;

DIRECT_EXPORT void* direct_create(void) {
  if (get_num_target() != 1 || N_TARGET * MAX_N_CLASS != 1 ||
      strcmp(get_threshold_type(), "float32") != 0 ||
      strcmp(get_leaf_output_type(), "float32") != 0) {
    return NULL;
  }
  DirectContext* ctx = (DirectContext*)malloc(sizeof(DirectContext));
  if (!ctx) return NULL;
  ctx->num_feature = (size_t)get_num_feature();
  ctx->entries = (union Entry*)malloc(ctx->num_feature * sizeof(union Entry));
  if (!ctx->entries) {
    free(ctx);
    return NULL;
  }
  return ctx;
}

DIRECT_EXPORT void direct_free(void* handle) {
  DirectContext* ctx = (DirectContext*)handle;
  if (ctx) {
    free(ctx->entries);
    free(ctx);
  }
}

/* No per-call allocation. Copy each NEW row into reusable Entry storage.
 * The caller owns output with capacity >= nrow and serializes calls. */
DIRECT_EXPORT void direct_predict(void* handle, const float* input,
                                  size_t nrow, float* output) {
  DirectContext* ctx = (DirectContext*)handle;
  for (size_t row = 0; row < nrow; ++row) {
    for (size_t col = 0; col < ctx->num_feature; ++col) {
      const float value = input[row * ctx->num_feature + col];
      if (isnan(value)) ctx->entries[col].missing = -1;
      else ctx->entries[col].fvalue = value;
    }
    /* Generated predict() ADDS tree scores to the caller's output. */
    output[row] = 0.0f;
    predict(ctx->entries, 0, &output[row]);
  }
}
