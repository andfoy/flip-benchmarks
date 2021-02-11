
#include <torch/torch.h>
#include <ATen/native/cpu/Loops.h>

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
 * Precompute indices used to perform flipping in parallel.
 * \param indices a 1D tensor containing all the indices to flip.
 * \param dims a list containing the reference dimensions of the tensor to flip.
 * \param flip_dims a list containing the index positions to flip.
 * \returns a 1D tensor containing the flipped indices.
 **/
torch::Tensor build_indices(torch::Tensor indices, torch::IntArrayRef dims, torch::IntArrayRef flip_dims) {
    torch::TensorIteratorConfig config;
    torch::Tensor output_tensor;
    auto iter =
        config.check_all_same_dtype(false)
            .declare_static_dtype_and_device(
                indices.scalar_type(),
                indices.device())
            .add_output(output_tensor)
            .add_input(indices)
            .build();

    torch::native::cpu_kernel(iter, [&](int64_t idx) -> int64_t {
        auto multi_dims = index_to_multidimensional(idx, dims);
        auto flipped_dims = flip_dimensions(multi_dims, dims, flip_dims);
        auto flipped_idx = multidimensional_to_index(flipped_dims, dims);
        return flipped_idx;
    });

    auto start = std::chrono::steady_clock::now();
    auto output = iter.output();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Flipped indices Elapsed time (ms): " << elapsed_seconds.count() * 1000 << std::endl;
    return output;
}

/**
 * Flip a N-dimensional tensor around a given set of dimensions.
 * \param input the input tensor to flip
 * \param flip_dims a list of dimensions used to flip the tensor.
 **/
torch::Tensor generalized_flip(torch::Tensor input, torch::IntArrayRef flip_dims) {
    auto sizes = input.sizes();
    auto start = std::chrono::steady_clock::now();
    auto indices = torch::arange(input.numel());
    auto indices_end = std::chrono::steady_clock::now();

    auto start_flip = std::chrono::steady_clock::now();
    auto flipped_indices = build_indices(indices, input.sizes(), flip_dims);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::chrono::duration<double> flip_elapsed_seconds = end - start_flip;
    std::chrono::duration<double> arange_elapsed_seconds = indices_end - start;
    std::cout << "Arange Elapsed time (ms): " << arange_elapsed_seconds.count() * 1000 << std::endl;
    std::cout << "Flip indices Elapsed time (ms): " << flip_elapsed_seconds.count() * 1000 << std::endl;
    std::cout << "Indices Elapsed time (ms): " << elapsed_seconds.count() * 1000 << std::endl;

    // std::cout << "Flipped indices size: " << flipped_indices.sizes() << "\n";
    // std::cout << "Indices size: " << indices.sizes() << "\n";

    torch::TensorIteratorConfig config;
    torch::Tensor output_tensor;
    auto iter =
        config.check_all_same_dtype(false)
            .declare_static_dtype_and_device(
                input.scalar_type(),
                input.device())
            .add_output(output_tensor)
            .add_input(input)
            .add_input(indices.as_strided(input.sizes(), input.strides()))
            .add_input(flipped_indices.as_strided(input.sizes(), input.strides()))
            .build();

    using scalar_t = uint8_t;
    // using Vec = torch::vec256::Vec256<scalar_t>;
    auto loop = [&](char **data, const int64_t *strides, int64_t n) {
        scalar_t *dst = (scalar_t *)data[0];
        scalar_t *src = (scalar_t *)data[1];
        int64_t *indices = (int64_t*)data[2];
        int64_t *flipped_indices = (int64_t*)data[3];

        int64_t base_index = indices[0];
        for (int64_t i = 0; i < n; i++)
        {
            auto flip_index = flipped_indices[i];
            auto cur_index = indices[i];
            *(dst - base_index + cur_index) = *(src - base_index + flip_index);
        }
    };

    // iter.serial_for_each(loop, {0, iter.numel()});
    start = std::chrono::steady_clock::now();
    iter.for_each(loop);
    auto output = iter.output();
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "For each Elapsed time (ms): " << elapsed_seconds.count() * 1000 << std::endl;
    return output;
}
