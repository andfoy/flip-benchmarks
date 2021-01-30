
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

    // Manual flipping comparison
    auto vertical_flip_torch = torch::flip(reshaped_tensor, {2});
    std::cout << "Vertical flip: " << vertical_flip_torch << "\n\n";

    auto output = vertical_flip(reshaped_tensor);
    std::cout << "TensorIterator flip: " << output << "\n";

    auto horizontal_flip_torch = torch::flip(reshaped_tensor, {3});
    std::cout << "Horizontal flip: " << horizontal_flip_torch << "\n\n";

    auto hz_flip = horizontal_flip(reshaped_tensor);
    std::cout << "TensorIterator flip: " << hz_flip << "\n";

    auto channel_flip_torch = torch::flip(reshaped_tensor, {1});
    std::cout << "Channel flip: " << channel_flip_torch << "\n\n";

    auto ch_flip = channel_flip(reshaped_tensor);
    std::cout << "TensorIterator flip: " << ch_flip << "\n";
    return 0;
}
