#pragma once
#include <dlfcn.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

namespace xgbfast {
// Load only trusted model libraries. Each worker owns its own Model.
class Model {
    void* library_ = nullptr;
    void* context_ = nullptr;
    void (*free_)(void*) = nullptr;
    int (*predict_)(void*, const float*, size_t, size_t, float*, size_t) = nullptr;
    size_t features_ = 0;
    std::thread::id owner_ = std::this_thread::get_id();
    template<class T> T symbol(const char* name) {
        dlerror();
        void* address = dlsym(library_, name);
        const char* error = dlerror();
        if (error) throw std::runtime_error(error);
        return reinterpret_cast<T>(address);
    }
    void check() const {
        if (owner_ != std::this_thread::get_id()) throw std::runtime_error("Model belongs to another worker");
        if (!context_) throw std::runtime_error("Model is closed");
    }
    void cleanup() noexcept {
        if (context_) free_(context_);
        context_ = nullptr;
        if (library_) dlclose(library_);
        library_ = nullptr;
    }
public:
    explicit Model(const std::string& path) {
        library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library_) throw std::runtime_error(dlerror());
        try {
            if (symbol<uint32_t(*)()>("xf_abi_version")() != 1) throw std::runtime_error("Unsupported model ABI");
            auto count = symbol<int32_t(*)()>("xf_num_features")();
            if (count < 1) throw std::runtime_error("Invalid feature count");
            features_ = static_cast<size_t>(count);
            free_ = symbol<void(*)(void*)>("xf_free");
            predict_ = symbol<decltype(predict_)>("xf_predict");
            context_ = symbol<void*(*)()>("xf_create")();
            if (!context_) throw std::runtime_error("Cannot allocate model context");
        } catch (...) { cleanup(); throw; }
    }
    ~Model() { cleanup(); }
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    size_t num_features() const noexcept { return features_; }
    void close() {
        if (owner_ != std::this_thread::get_id()) throw std::runtime_error("Close on the owning worker");
        cleanup();
    }
    float predict(std::span<const float> input) {
        check();
        if (input.size() != features_ || !input.data()) throw std::invalid_argument("Wrong feature count or null input");
        if (reinterpret_cast<uintptr_t>(input.data()) % alignof(float)) throw std::invalid_argument("Unaligned input");
        float output = 0;
        int status = predict_(context_, input.data(), 1, features_, &output, 1);
        if (status == 2) throw std::invalid_argument("Infinity is not supported; use NaN for missing values");
        if (status) throw std::runtime_error("Native prediction failed");
        return output;
    }
};
}
