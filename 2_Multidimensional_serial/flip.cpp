
#include <torch/torch.h>

/**
 * Convert a 1D index into an N-D one.
 *
 * \param idx a one dimensional index to convert
 * \param dims a list containing the reference N-dimensional bases.
 * \return the corresponding index in the N-dimensional reference in the form
 * of a vector.
 **/
std::vector<int64_t> index_to_multidimensional(int64_t idx, torch::IntArrayRef dims) {
    const int64_t size = dims.size();
    std::vector<int64_t> result;
    int64_t off = idx;
    int64_t stride = 1;
    for(int64_t i = size - 1; i > 0; i--) {
        auto dim = dims[i];
        int64_t next_stride = dim * stride;
        int64_t mod = off % next_stride;
        off -= mod;
        int64_t div = mod / stride;
        result.insert(result.begin(), div);
        stride = next_stride;
    }
    int64_t final_dim = off / stride;
    result.insert(result.begin(), final_dim);
    return result;
}

/**
 * Convert an N-dimensional index into a 1-dimensional one.
 * \param multidim a vector containing the N-dimensional coordinates to convert.
 * \param dims a list containing the reference N-dimensional bases.
 * \return the corresponding index in a 1-dimensional reference.
 **/
int64_t multidimensional_to_index(std::vector<int64_t> multidim, torch::IntArrayRef dims) {
    int64_t idx = 0;
    int64_t stride = 1;

    for(int64_t i = dims.size() - 1; i > 0; i--) {
        idx += multidim[i] * stride;
        stride *= dims[i];
    }
    idx += multidim[0] * stride;
    return idx;
}

/**
 * Flip a N-dimensional coordinate, given a set of dimensions to flip and the N-dimensional reference bases.
 * \param multidim a vector containing the N-dimensional coordinates to flip.
 * \param dims a list containing the reference N-dimensional bases.
 * \param flip_dims a list containing the index positions to flip.
 * \return a vector containing the flipped N-dimensional coordinates.
 **/
std::vector<int64_t> flip_dimensions(std::vector<int64_t> multidim, torch::IntArrayRef dims, torch::IntArrayRef flip_dims) {
    for(int64_t dim : flip_dims) {
        multidim[dim] = dims[dim] - multidim[dim] - 1;
    }
    return multidim;
}

/**
 * Flip a N-dimensional tensor around a given set of dimensions.
 * \param input the input tensor to flip
 * \param flip_dims a list of dimensions used to flip the tensor.
 **/
torch::Tensor generalized_flip(torch::Tensor input, torch::IntArrayRef flip_dims) {
    auto sizes = input.sizes();
    torch::TensorIteratorConfig config;
    torch::Tensor output_tensor;
    auto iter =
        config.check_all_same_dtype(false)
            .declare_static_dtype_and_device(
                input.scalar_type(),
                input.device())
            .add_output(output_tensor)
            .add_input(input)
            // .add_input()
            .build();

    using scalar_t = uint8_t;
    // using Vec = torch::vec256::Vec256<scalar_t>;
    auto loop = [&](char **data, const int64_t *strides, int64_t n) {
        scalar_t *dst = (scalar_t *)data[0];
        scalar_t *src = (scalar_t *)data[1];

        for (int64_t i = 0; i < n; i++)
        {
            auto multi_dims = index_to_multidimensional(i, sizes);
            auto flipped_dims = flip_dimensions(multi_dims, sizes, flip_dims);
            auto flipped_idx = multidimensional_to_index(flipped_dims, sizes);
            dst[flipped_idx] = src[i];
        }
    };

    iter.serial_for_each(loop, {0, iter.numel()});
    auto output = iter.output();
    return output;
}
