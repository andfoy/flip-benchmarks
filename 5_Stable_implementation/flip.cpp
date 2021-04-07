
#include <torch/torch.h>
// #include <ATen/native/cpu/Loops.h>
// #include <ATen/native/TensorAdvancedIndexing.h>

struct Indexer {
  Indexer(int64_t num_indexers, char** indexers, const int64_t* indexer_strides,
          torch::IntArrayRef original_sizes, torch::IntArrayRef original_strides)
    : num_indexers(num_indexers)
    , indexers(indexers)
    , indexer_strides(indexer_strides)
    , original_strides(original_strides.data())
    , original_sizes(original_sizes.data()) {
    AT_ASSERT(original_strides.size() == num_indexers);
    AT_ASSERT(original_sizes.size() == num_indexers);
  }

  int64_t num_indexers;
  char** indexers;
  const int64_t* indexer_strides;
  const int64_t* original_strides;
  const int64_t* original_sizes;

  int64_t get(int64_t idx) {
    int64_t offset = 0;
    for (int j = 0; j < num_indexers; j++) {
      int64_t value = *(int64_t*)&indexers[j][idx * indexer_strides[j]];
      offset += value * original_strides[j];
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
  for(auto dim: flip_dims) {
    auto dim_size = input.size(dim);
    auto index = build_index(input.ndimension(), dim, dim_size);
    auto stride = input.stride(dim);
    // std::cout << "Dim: " << dim << ", Size: " << dim_size << ", Stride: " << stride << std::endl;
    auto input_index_ptr = index.data_ptr<int64_t>();
    for(int64_t i = 0; i < dim_size; i++) {
      input_index_ptr[i] = static_cast<int64_t>(dim_size - i - 1); // * stride;
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

void index_kernel(torch::TensorIterator& iter, torch::IntArrayRef index_size,
                  torch::IntArrayRef index_stride, bool serial_execution=false) {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(torch::ScalarType::Half, torch::ScalarType::Bool, torch::ScalarType::BFloat16,
      iter.dtype(), "flip_cpu", [&] {
        int ntensor = iter.ntensors();
        // When launch the index parallel version, set a relative samll grain size less than the INTERNAL::GRAIN_SIZE
        // to make the whole available thread numbers get more balanced work load and a better cache location.
        // The grain size here is chosen by the op benchmark to overcome the thread launch overhead
        const int index_parallel_grain_size = 3000;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
            auto indexer = Indexer(ntensor - 2, &data[2], &strides[2], index_size, index_stride);
            char* dst = data[0];
            char* src = data[1];

            // int64_t offset = indexer.get(0);

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
    torch::DimVector indexed_sizes;
    torch::DimVector indexed_strides;
    int64_t element_size_bytes = input.element_size();

    // Set stride to zero on the dimensions that are going to be flipped
    for(auto dim: dims) {
      strides[dim] = 0;
      indexed_sizes.push_back(input.size(dim));
      indexed_strides.push_back(input.stride(dim) * element_size_bytes);
    }

    // Restride the input to index only on the dimensions to flip
    auto restrided_input = input.as_strided(shape, strides);
    // auto indices = build_indices(input, dims, strides)

    std::vector<torch::Tensor> indices;
    std::vector<int64_t> transposed_indices;
    // std::vector<int64_t> sizes;
    indices = build_indices_loop(input, dims);

    auto iter = make_index_iterator(restrided_input, indices);
    index_kernel(iter, indexed_sizes, indexed_strides);
    auto result = iter.output();

    return result;
}

TORCH_LIBRARY(flip_ops, m) {
  m.def("flip", generalized_flip);
}
