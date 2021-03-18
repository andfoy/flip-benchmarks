
#include <iostream>
#include <string>

#include <torch/torch.h>

#include "flip.h"

int main()
{
    int64_t dim0 = 5;
    int64_t dim1 = 3; // C
    int64_t dim2 = 4; // H
    int64_t dim3 = 6; // W
    auto opts = torch::TensorOptions(torch::kCPU).dtype(torch::kUInt8);
    auto tensor = torch::arange(dim0 * dim1 * dim2 * dim3, opts);
    auto reshaped_tensor = tensor.reshape({dim0, dim1, dim2, dim3}).contiguous();
    std::cout << "Input: " << reshaped_tensor << "\n\n";

    std::cout << "--------------- GENERALIZED FLIP -----------------" << std::endl << std::endl;

    auto start = std::chrono::steady_clock::now();
    auto gen_multiflip = generalized_flip(reshaped_tensor, {1, 3});
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time (ms): " << elapsed_seconds.count() * 1000 << std::endl;

    std::cout << "--------------- GENERALIZED FLIP -----------------" << std::endl << std::endl;

    std::cout << "--------------- ADV INDEXING FLIP ----------------" << std::endl << std::endl;

    torch::optional<torch::Tensor> dim0_dims;
    torch::optional<torch::Tensor> dim1_dims = torch::arange(dim1 - 1, -1, -1).unsqueeze(-1);
    torch::optional<torch::Tensor> dim2_dims; // = torch::arange(dim2 - 1, -1, -1).unsqueeze(0).unsqueeze(-1);
    torch::optional<torch::Tensor> dim3_dims = torch::arange(dim3 - 1, -1, -1).unsqueeze(0);

    std::cout << "Dim1: " << dim1_dims.value() << std::endl;
    // std::cout << "Dim2: " << dim2_dims.value() << std::endl;
    std::cout << "Dim3: " << dim3_dims.value() << std::endl;

    start = std::chrono::steady_clock::now();
    auto index_flip = torch::index(reshaped_tensor, {dim0_dims, dim1_dims, dim2_dims, dim3_dims});
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Elapsed time (ms): " << elapsed_seconds.count() * 1000 << std::endl;

    std::cout << "--------------- ADV INDEXING FLIP ----------------" << std::endl << std::endl;

    std::cout << "--------------------- FLIP -----------------------" << std::endl << std::endl;

    start = std::chrono::steady_clock::now();
    auto multiflip = torch::flip(reshaped_tensor, {1, 3});
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Elapsed time (ms): " << elapsed_seconds.count() * 1000 << std::endl;
    // std::cout << "Multiflip: " << multiflip << "\n";
    std::cout << "Generalized size: " << gen_multiflip.sizes() << std::endl;
    std::cout << "Advanced Indexing size: " << index_flip.sizes() << std::endl;
    std::cout << "torch::flip size: " << multiflip.sizes() << std::endl;

    std::cout << "--------------------- FLIP -----------------------" << std::endl << std::endl;

    if(!multiflip.allclose(index_flip)) {
        std::cout << "Error! Indexing implementation values differ!" << "\n";
    }
    else if(!multiflip.allclose(gen_multiflip)) {
        std::cout << "Error! Generalized implementation values differ!" << "\n";
    }
    else {
        std::cout << "Values are the same!" << "\n";
    }
    return 0;
}
