CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Wpedantic
SANFLAGS := -std=c++20 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -Wall -Wextra

.PHONY: all test test-san bench clean
all: example fuzz_side_index fuzz_order_side bench_side_index

example: example.cpp side_index.hpp order_side.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

fuzz_side_index: fuzz_side_index.cpp side_index.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

fuzz_side_index_san: fuzz_side_index.cpp side_index.hpp
	$(CXX) $(SANFLAGS) -o $@ $<

fuzz_order_side: fuzz_order_side.cpp side_index.hpp order_side.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

fuzz_order_side_san: fuzz_order_side.cpp side_index.hpp order_side.hpp
	$(CXX) $(SANFLAGS) -o $@ $<

bench_side_index: bench_side_index.cpp side_index.hpp
	$(CXX) $(CXXFLAGS) -O3 -march=native -DNDEBUG -o $@ $<

test: fuzz_side_index fuzz_order_side
	./fuzz_side_index
	./fuzz_order_side

# AddressSanitizer + UBSan，含 assert
test-san: fuzz_side_index_san fuzz_order_side_san
	./fuzz_side_index_san 5000 20
	./fuzz_order_side_san 3000 10

bench: bench_side_index
	./bench_side_index

clean:
	rm -f example fuzz_side_index fuzz_side_index_san fuzz_order_side fuzz_order_side_san bench_side_index
