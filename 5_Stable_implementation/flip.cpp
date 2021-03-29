
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
      offset += value * original_strides[j];
    }
    return offset;
  }
};

static std::string shapes_as_str(torch::TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
}

static bool hasContiguousSubspace(torch::TensorList tl) {
  // true if all the non-null tensors are adjacent
  auto isDefined = [](const torch::Tensor & tensor){ return tensor.defined(); };
  auto isNull = [](const torch::Tensor & tensor){ return !tensor.defined(); };
  auto start = std::find_if(tl.begin(), tl.end(), isDefined);
  auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
  auto it = std::find_if(start, stop.base(), isNull);
  return it == stop.base();
}

// Transposes the tensor and indices together so that all the non-null indices
// index the first k dimensions of the tensor. Returns the transposed tensor
// and the reordered indices. For example:
// transposeToFront(tensor, {nullptr, a, nullptr, b})
// returns
// tensor.permute([1, 3, 0, 2]), {a, b, nullptr, nullptr}
static std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<int64_t>>
transposeToFront(torch::Tensor self, torch::TensorList indices) {
  std::vector<int64_t> dims;
  std::vector<int64_t> inv_dims(self.dim());
  std::vector<torch::Tensor> transposedIndices;

  dims.reserve(self.dim());
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (indices[i].defined()) {
      inv_dims[i] = dims.size();
      dims.push_back(i);
      transposedIndices.emplace_back(indices[i]);
    }
  }
  for (auto i = decltype(self.dim()){0}; i < self.dim(); i++) {
    if (!indices[i].defined()) {
      inv_dims[i] = dims.size();
      dims.push_back(i);
      transposedIndices.emplace_back();
    }
  }
  auto permuted_input = self.permute(dims);

  // std::reverse(dims.begin(), dims.end());
  return std::make_tuple(permuted_input, std::move(transposedIndices), inv_dims);
}

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


torch::Tensor create_index(int64_t dim_pos, int64_t dim, size_t num_dims, torch::Tensor input) {
    auto dim_index = torch::arange(input.size(dim) - 1, -1, -1).to(
        input.device());
    for(int64_t i = 0; i < dim_pos; i++) {
        dim_index = dim_index.unsqueeze(0);
    }
    for(int64_t i = dim_pos + 1; i < num_dims; i++) {
        dim_index = dim_index.unsqueeze(-1);
    }
    return dim_index;  // dim_index.expand_as(input);
}


std::tuple<torch::Tensor, std::vector<torch::Tensor>, std::vector<int64_t>>
build_indices(torch::Tensor input, torch::IntArrayRef flip_dims) {
    std::vector<torch::Tensor> indices;

    const int64_t* dim_ptr = flip_dims.begin();
    int64_t dim_pos = 0;
    for(int64_t i = 0; i < input.dim(); i++) {
      if(dim_pos < flip_dims.size() && i == *dim_ptr) {
        auto index = create_index(dim_pos, *dim_ptr, flip_dims.size(), input);
        indices.push_back(index);
        dim_ptr++;
        dim_pos++;
      } else {
        indices.emplace_back();
      }
    }

    try {
      indices = expand_outplace(indices);
    } catch (std::exception& e) {
      TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together"
                        " with shapes ", shapes_as_str(indices));
    }

    // add missing null Tensors so that it matches self.dim()
    while (indices.size() < (size_t) input.dim()) {
      indices.emplace_back();
    }

    // if the non-null indices are not all adjacent, transpose self and indices
    // together so that they're adjacent at the front
    std::vector<int64_t> transposed_indices;
    if (!hasContiguousSubspace(indices)) {
      std::tie(input, indices, transposed_indices) = transposeToFront(
        input, indices);
    }

    // Ensure indices are on the same device as self
    for (size_t i = 0; i < indices.size(); i++) {
      if (indices[i].defined() && indices[i].device() != input.device()) {
        indices[i] = indices[i].to(input.device());
      }
    }

    return std::make_tuple(input, std::move(indices), transposed_indices);
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

    std::vector<torch::Tensor> indices;
    std::vector<int64_t> transposed_indices;
    // std::vector<int64_t> sizes;
    std::tie(input, indices, transposed_indices) = build_indices(input, dims);

    auto advanced_index = AdvancedIndex(input, indices);
    auto iter = make_index_iterator(advanced_index);
    index_kernel(iter, advanced_index.indexed_sizes, advanced_index.indexed_strides);
    auto result = iter.output();

    if(transposed_indices.size() > 0) {
      result = result.permute(transposed_indices);
    }

    return result;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("flip", generalized_flip);
}
