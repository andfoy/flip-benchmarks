
#include <iostream>
#include <string>

#include <torch/torch.h>

#include "flip.h"


std::vector<std::vector<int64_t>>
power_set(size_t idx, size_t index_size,
          std::vector<std::vector<int64_t>> result, std::vector<int64_t> tail = {}) {
    if(idx == index_size) {
        return result;
    }

    tail.push_back(idx);

    if (std::find(result.begin(), result.end(), tail) == result.end()) {
        result.push_back(tail);
    }

    for(size_t idx_copy = idx + 1; idx_copy < index_size; idx_copy++) {
        std::vector<int64_t> tail_copy(tail);
        result = power_set(idx_copy, index_size, result, tail_copy);
    }

    return power_set(idx + 1, index_size, result);

}

std::vector<std::vector<int64_t>>
power_set(std::vector<int64_t> indices) {
    std::vector<std::vector<int64_t>> result;
    size_t size = indices.size();
    return power_set(0, size, result);
}

torch::Tensor call_generalized_flip(torch::Tensor input, std::vector<int64_t> dims) {
    auto start = std::chrono::steady_clock::now();
    auto gen_multiflip = generalized_flip(input, dims);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Generalized flip elapsed time (ms): " << elapsed_seconds.count() * 1000 << std::endl;
    return gen_multiflip;
}

torch::Tensor call_torch_flip(torch::Tensor input, std::vector<int64_t> dims) {
    auto start = std::chrono::steady_clock::now();
    auto flip = torch::flip(input, dims);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "torch::flip elapsed time (ms): " << elapsed_seconds.count() * 1000 << std::endl;
    return flip;
}


int main()
{
    int64_t dim0 = 5;
    int64_t dim1 = 3; // C
    int64_t dim2 = 4; // H
    int64_t dim3 = 6; // W

    auto dims_sets = power_set({dim0, dim1, dim2, dim3});

    auto opts = torch::TensorOptions(torch::kCPU).dtype(torch::kUInt8);
    auto tensor = torch::arange(dim0 * dim1 * dim2 * dim3, opts);
    auto reshaped_tensor = tensor.reshape({dim0, dim1, dim2, dim3}).contiguous();
    auto types = {torch::kInt32, torch::kInt64, torch::kByte, torch::kFloat32, torch::kFloat64};
    auto contiguous = {true, false};
    auto channels_last = {true, false};

    for(auto type: types) {
        reshaped_tensor = reshaped_tensor.to(type);
        for(auto is_contiguous: contiguous) {
            auto test_tensor = reshaped_tensor;
            if(!is_contiguous) {
                test_tensor = test_tensor.transpose(0, 1);
            }
            for(auto is_channels_last: channels_last) {
                if(is_channels_last) {
                    test_tensor = test_tensor.to(torch::MemoryFormat::ChannelsLast);
                } else {
                    test_tensor = test_tensor.to(torch::MemoryFormat::Contiguous);
                }
                for(auto dim_set : dims_sets) {
                    std::cout << "Dimensions to flip: " << dim_set << std::endl;
                    std::cout << "Type: " << type << std::endl;
                    std::cout << "Contiguous: " << test_tensor.is_contiguous() << std::endl;
                    std::cout << "Memory format: " << test_tensor.suggest_memory_format() << std::endl;
                    auto gen_flip = call_generalized_flip(test_tensor, dim_set);
                    auto torch_flip = call_torch_flip(test_tensor, dim_set);
                    std::cout << "Generalized size: " << gen_flip.sizes() << std::endl;
                    std::cout << "torch::flip size: " << torch_flip.sizes() << std::endl;
                    if(!torch_flip.allclose(gen_flip)) {
                        std::cout << "Error! Generalized implementation values differ!" << "\n";
                        return 2;
                    }
                    else {
                        std::cout << "Values are the same!" << "\n";
                    }
                    std::cout << "----------------------------------------" << std::endl;
                }
            }
        }
    }
    return 0;
}
