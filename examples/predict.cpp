#include "xgbfast.hpp"
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "Usage: predict path/to/model.so\n"; return 1; }
    try {
        xgbfast::Model model(argv[1]);
        std::vector<float> features(model.num_features(), 0.0f);
        std::cout << model.predict(features) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
