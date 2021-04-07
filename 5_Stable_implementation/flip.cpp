
#include <torch/torch.h>

struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides)
    : num_indexers(num_indexers)
    , indexers(indexers)
    , indexer_strides(indexer_strides) {

  }

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;

  int64_t get(int64_t idx) {
    int64_t offset = 0;
    for (int j = 0; j < num_indexers; j++) {
      int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]];
      offset += value;
    }
    return offset;
  }
};

torch::Tensor build_index(int64_t num_dims, int64_t flip_dim, int64_t dim_size) {
  auto new_shape = std::vector<int64_t>(num_dims, 1);
  new_shape[flip_dim] = dim_size;
  return torch::empty(new_shape).to(torch::kInt64);
}

std::vector<torch::Tensor> build_indices_loop(torch::Tensor input, torch::IntArrayRef flip_dims) {
  std::vector<torch::Tensor> indices;
  int64_t element_size_bytes = input.element_size();
  for(auto dim: flip_dims) {
    auto dim_size = input.size(dim);
    auto index = build_index(input.ndimension(), dim, dim_size);
    auto stride = input.stride(dim);
    auto input_index_ptr = index.data_ptr<int64_t>();

    for(int64_t i = 0; i < dim_size; i++) {
      input_index_ptr[i] = static_cast<int64_t>(dim_size - i - 1) * stride * element_size_bytes;
    }
    indices.push_back(index);
  }
  return indices;
}

static torch::TensorIterator make_index_iterator(const torch::Tensor input, const std::vector<torch::Tensor> indices) {
  torch::TensorIteratorConfig config;
  config.set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .declare_static_dtype_and_device(input.scalar_type(), input.device())
        .add_output(torch::Tensor())
        .add_input(input);
  for (auto& index : indices) {
    config.add_input(index);
  }
  return config.build();
}

void index_kernel(torch::TensorIterator& iter, bool serial_execution=false) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(torch::ScalarType::Half, torch::ScalarType::Bool, torch::ScalarType::BFloat16,
      iter.dtype(), "flip_cpu", [&] {
        int ntensor = iter.ntensors();
        // When launch the index parallel version, set a relative samll grain size less than the INTERNAL::GRAIN_SIZE
        // to make the whole available thread numbers get more balanced work load and a better cache location.
        // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
        const int index_parallel_grain_size = 3000;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
            auto indexer = Indexer(ntensor - 2, &data[2], &strides[2]);
            char* dst = data[0];
            char* src = data[1];

            for (int64_t i = 0; i < n; i++) {
                int64_t offset = indexer.get(i);
                *(scalar_t*)(dst + strides[0] * i) = *(scalar_t*)(src + strides[1] * i + offset);
            }
        };

        if (serial_execution) {
            iter.serial_for_each(loop, {0, iter.numel()});
        } else {
            iter.for_each(loop, index_parallel_grain_size);
        }
      });
}

/**
 * Flip a N-dimensional tensor around a given set of dimensions.
 * \param input the input tensor to flip
 * \param flip_dims a list of dimensions used to flip the tensor.
 **/
torch::Tensor generalized_flip(torch::Tensor input, torch::IntArrayRef flip_dims) {
    // std::sort(flip_dims.begin(), flip_dims.end());
    std::vector<int64_t> dims;
    for(int64_t dim: flip_dims) {
        dims.push_back(dim);
    }

    std::sort(dims.begin(), dims.end());

    auto shape = input.sizes().vec();
    auto strides = input.strides().vec();
    int64_t element_size_bytes = input.element_size();

    // Set stride to zero on the dimensions that are going to be flipped
    for(auto dim: dims) {
      strides[dim] = 0;
    }

    // Restride the input to index only on the dimensions to flip
    auto restrided_input = input.as_strided(shape, strides);
    auto indices = build_indices_loop(input, dims);
    auto iter = make_index_iterator(restrided_input, indices);
    index_kernel(iter);

    auto result = iter.output();
    return result;
}

TORCH_LIBRARY(flip_ops, m) {
  m.def("flip", generalized_flip);
}
