
#include <torch/torch.h>

torch::Tensor vertical_flip(torch::Tensor input);
torch::Tensor horizontal_flip(torch::Tensor input);
torch::Tensor channel_flip(torch::Tensor input);
torch::Tensor generalized_flip(torch::Tensor input, torch::IntArrayRef flip_dims);
