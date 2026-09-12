#ifndef XGBFAST_H
#define XGBFAST_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI 1. One context per concurrent caller; no allocation during prediction.
 * Dense row-major native float32. NaN = missing; infinity is rejected.
 * Input/output must not overlap. Status: 0 success, 1 invalid bounds, 2 infinity.
 * On failure output may be partially written. */
uint32_t xf_abi_version(void);
int32_t xf_num_features(void);
void* xf_create(void);
void xf_free(void*);
int xf_predict(void*, const float*, size_t, size_t, float*, size_t);
#ifdef __cplusplus
}
#endif
#endif
