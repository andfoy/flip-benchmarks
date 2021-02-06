
#include <iostream>
#include <string>

#include <torch/torch.h>

#include "flip.h"

int main()
{
    int64_t dim1 = 3; // C
    int64_t dim2 = 4; // H
    int64_t dim3 = 3; // W
    auto opts = torch::TensorOptions(torch::kCPU).dtype(torch::kUInt8);
    auto tensor = torch::arange(dim1 * dim2 * dim3, opts);
    auto reshaped_tensor = tensor.reshape({1, dim1, dim2, dim3}).contiguous();
    std::cout << "Input: " << reshaped_tensor << "\n\n";

    auto hz_gen_flip = generalized_flip(reshaped_tensor, {1, 2, 3});
    std::cout << "Gen TensorIterator flip: " << hz_gen_flip << "\n";

    auto multiflip = torch::flip(reshaped_tensor, {1, 2, 3});
    std::cout << "Multiflip: " << multiflip << "\n";
    return 0;
}
