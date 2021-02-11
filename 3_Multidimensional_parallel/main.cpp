
#include <iostream>
#include <string>

#include <torch/torch.h>

#include "flip.h"

int main()
{
    int64_t dim0 = 5;
    int64_t dim1 = 3; // C
    int64_t dim2 = 800; // H
    int64_t dim3 = 600; // W
    auto opts = torch::TensorOptions(torch::kCPU).dtype(torch::kUInt8);
    auto tensor = torch::arange(dim0 * dim1 * dim2 * dim3, opts);
    auto reshaped_tensor = tensor.reshape({dim0, dim1, dim2, dim3}).contiguous();
    // std::cout << "Input: " << reshaped_tensor << "\n\n";
    auto start = std::chrono::steady_clock::now();
    auto gen_multiflip = generalized_flip(reshaped_tensor, {1, 2, 3});
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time (ms): " << elapsed_seconds.count() * 1000 << std::endl;

    std::cout << "Flipped!\n";
    // std::cout << "Gen TensorIterator flip: " << gen_multiflip << "\n";

    start = std::chrono::steady_clock::now();
    auto multiflip = torch::flip(reshaped_tensor, {1, 2, 3});
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Elapsed time (ms): " << elapsed_seconds.count() * 1000 << std::endl;
    // std::cout << "Multiflip: " << multiflip << "\n";

    if(!multiflip.allclose(gen_multiflip)) {
        std::cout << "Error! Implementation values differ!" << "\n";
    } else {
        std::cout << "Values are the same!" << "\n";
    }
    return 0;
}
