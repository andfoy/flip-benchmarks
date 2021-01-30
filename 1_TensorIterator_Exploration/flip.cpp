
#include <torch/torch.h>

torch::Tensor vertical_flip(torch::Tensor input)
{
    // Assuming BCHW layout
    int64_t dim1 = input.size(1);
    int64_t dim2 = input.size(2);
    int64_t dim3 = input.size(3);

    torch::TensorIteratorConfig config;
    torch::Tensor output_tensor;
    auto iter =
        config.check_all_same_dtype(false)
            .declare_static_dtype_and_device(
                input.scalar_type(),
                input.device())
            .add_output(output_tensor)
            .add_input(input)
            .build();

    using scalar_t = uint8_t;
    // using Vec = torch::vec256::Vec256<scalar_t>;
    auto loop = [&](char **data, const int64_t *strides, int64_t n) {
        scalar_t *dst = (scalar_t *)data[0];
        scalar_t *src = (scalar_t *)data[1];
        int64_t stride = 0;
        int64_t ch_stride = 0;
        for (int64_t i = 0; i < n; i++)
        {
            if (i % (dim2 * dim3) == 0)
            {
                ch_stride++;
                stride = 0;
            }
            if (i % dim3 == 0)
            {
                stride++;
            }

            int64_t flip_idx = ((ch_stride * dim2 * dim3 - (stride * dim3)) + (i % dim3));
            dst[i] = src[flip_idx];
        }
    };

    iter.serial_for_each(loop, {0, iter.numel()});
    auto output = iter.output();
    return output;
}


torch::Tensor horizontal_flip(torch::Tensor input)
{
    // Assuming BCHW layout
    int64_t dim1 = input.size(1);
    int64_t dim2 = input.size(2);
    int64_t dim3 = input.size(3);

    torch::TensorIteratorConfig config;
    torch::Tensor output_tensor;
    auto iter =
        config.check_all_same_dtype(false)
            .declare_static_dtype_and_device(
                input.scalar_type(),
                input.device())
            .add_output(output_tensor)
            .add_input(input)
            .build();

    using scalar_t = uint8_t;
    // using Vec = torch::vec256::Vec256<scalar_t>;
    auto loop = [&](char **data, const int64_t *strides, int64_t n) {
        scalar_t *dst = (scalar_t *)data[0];
        scalar_t *src = (scalar_t *)data[1];
        int64_t stride = 0;
        for (int64_t i = 0; i < n; i++)
        {
            if (i % dim3 == 0)
            {
                stride++;
            }

            int64_t flip_idx = (stride * dim3) - ((i % dim3) + 1);
            dst[i] = src[flip_idx];
        }
    };

    iter.serial_for_each(loop, {0, iter.numel()});
    auto output = iter.output();
    return output;
}


torch::Tensor channel_flip(torch::Tensor input)
{
    // Assuming BCHW layout
    int64_t dim1 = input.size(1);
    int64_t dim2 = input.size(2);
    int64_t dim3 = input.size(3);

    torch::TensorIteratorConfig config;
    torch::Tensor output_tensor;
    auto iter =
        config.check_all_same_dtype(false)
            .declare_static_dtype_and_device(
                input.scalar_type(),
                input.device())
            .add_output(output_tensor)
            .add_input(input)
            .build();

    using scalar_t = uint8_t;
    // using Vec = torch::vec256::Vec256<scalar_t>;
    auto loop = [&](char **data, const int64_t *strides, int64_t n) {
        scalar_t *dst = (scalar_t *)data[0];
        scalar_t *src = (scalar_t *)data[1];
        int64_t stride = dim1;
        for (int64_t i = 0; i < n; i++)
        {
            if (i % (dim2 * dim1) == 0)
            {
                stride--;
            }

            int64_t flip_idx = (stride * dim2 * dim3) + (i % (dim2 * dim3));
            dst[i] = src[flip_idx];
        }
    };

    iter.serial_for_each(loop, {0, iter.numel()});
    auto output = iter.output();
    return output;
}
