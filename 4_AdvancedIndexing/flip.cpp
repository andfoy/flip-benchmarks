
#include <torch/torch.h>
// #include <ATen/native/cpu/Loops.h>
// #include <ATen/native/TensorAdvancedIndexing.h>


struct AdvancedIndex {
  AdvancedIndex(const torch::Tensor& src, torch::TensorList indices);

  torch::Tensor src;
  std::vector<torch::Tensor> indices;
  torch::DimVector indexed_sizes;
  torch::DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};

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
      int64_t size = original_sizes[j];
      if (value < -size || value >= size) {
        TORCH_CHECK_INDEX(false, "index ", value, " is out of bounds for dimension ", j, " with size ", size);
      }
      if (value < 0) {
        value += size;
      }
      offset += value * original_strides[j];
    }
    return offset;
  }
};

// Replace indexed dimensions in src with stride 0 and the size of the result tensor.
// The offset in these dimensions is computed by the kernel using the index tensor's
// values and the stride of src. The new shape is not meaningful. It's used to make
// the shape compatible with the result tensor.
static torch::Tensor restride_src(const torch::Tensor& src, int64_t dims_before, int64_t dims_indexed,
                                  torch::IntArrayRef replacement_shape) {
  auto shape = torch::DimVector(src.sizes());
  auto strides = torch::DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

// Add dimensions of size 1 to an index tensor so that it can be broadcast to the result
// shape and iterated over element-wise like the result tensor and the restrided src.
static torch::Tensor reshape_indexer(const torch::Tensor& index, int64_t dims_before, int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = torch::DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}


static bool all_strides_match(torch::TensorList tensors) {
  TORCH_CHECK(tensors.size() >= 1);
  auto strides = tensors[0].strides();
  for (auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

AdvancedIndex::AdvancedIndex(const torch::Tensor& src, torch::TensorList indices_list)
{
  int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  torch::IntArrayRef replacement_shape;
  for (size_t dim = 0; dim < indices_list.size(); dim++) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
    TORCH_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }

  // For CUDA tensors, force all index tensors to have the same striding to
  // simplify the CUDA kernel.
  if (indices.size() >= 2 && this->src.device().type() == torch::kCUDA) {
    if (!all_strides_match(indices)) {
      for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = indices[i].contiguous();
      }
    }
  }
}


torch::Tensor create_index(int64_t dim, torch::Tensor input) {
    auto dim_index = torch::arange(input.size(dim) - 1, -1, -1).to(
        input.device());
    for(int64_t i = 0; i < dim; i++) {
        dim_index = dim_index.unsqueeze(0);
    }
    for(int64_t i = dim + 1; i < input.dim(); i++) {
        dim_index = dim_index.unsqueeze(-1);
    }
    return dim_index.expand_as(input);
}


std::tuple<torch::Tensor, std::vector<torch::Tensor>> build_indices(torch::Tensor input, torch::IntArrayRef flip_dims) {
    std::vector<torch::Tensor> result;
    std::vector<torch::Tensor> empty_result;
    std::vector<int64_t> sizes;
    std::vector<int64_t> empty_sizes;

    const int64_t* dim_ptr = flip_dims.begin();

    // Create indices for each declared dimension and empty ones where there are not dimensions
    for(int64_t i = 0; i < input.dim(); i++) {
        auto dim = *dim_ptr;
        if(i == dim) {
            auto index = create_index(dim, input);
            result.push_back(index);
            sizes.push_back(i);
        } else {
            empty_result.emplace_back();
            empty_sizes.push_back(i);
        }
    }

    // Join both defined and empty indices in order to permute the input tensor
    result.insert(result.end(),
                  std::make_move_iterator(empty_result.begin()),
                  std::make_move_iterator(empty_result.end()));

    sizes.insert(sizes.end(),
                 std::make_move_iterator(empty_sizes.begin()),
                 std::make_move_iterator(empty_sizes.end()));

    return std::make_tuple(input.permute(sizes), std::move(result));
}

static torch::TensorIterator make_index_iterator(const AdvancedIndex& info) {
  torch::TensorIteratorConfig config;
  config.set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .declare_static_dtype_and_device(info.src.scalar_type(), info.src.device())
        .add_output(torch::Tensor())
        .add_input(info.src);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

void index_kernel(torch::TensorIterator& iter, torch::IntArrayRef index_size,
                  torch::IntArrayRef index_stride, bool serial_execution=false) {
    using scalar_t = uint8_t;
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

    std::vector<torch::Tensor> indices;
    std::tie(input, indices) = build_indices(input, dims);

    auto advanced_index = AdvancedIndex(input, indices);
    auto iter = make_index_iterator(advanced_index);
    index_kernel(iter, advanced_index.indexed_sizes, advanced_index.indexed_strides);
    return iter.output();
}
