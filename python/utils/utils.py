import numpy as np


def get_patch(input_array, i, j, kernel_width, kernel_height, stride):
    if len(input_array.shape) > 2:
        return input_array[:, i * stride:i * stride + kernel_width, j * stride:j * stride + kernel_height]
    return input_array[i * stride:i * stride + kernel_width, j * stride:j * stride + kernel_height]


def get_max_index(patch_array):
    max_value = patch_array[0, 0]
    i_max = 0
    j_max = 0
    for i in range(0, patch_array.shape[0]):
        for j in range(0, patch_array.shape[1]):
            if patch_array[i, j] > max_value:
                i_max = i
                j_max = j
                max_value = patch_array[i, j]
    return i_max, j_max


# do element wise operation to numpy array
def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


def conv(input_array,
         kernel_array,
         output_array,
         stride, bias):
    """
    calculate convolution for 2D 3D
    """
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                                     get_patch(input_array, i, j, kernel_width,
                                               kernel_height, stride) * kernel_array
                                 ).sum() + bias


def padding(input_array, zp):
    """
    add zero padding for 2D 3D
    """
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[:,
            zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array


def conv_matrix(input_array,
                kernel_array,
                kernel_width,
                kernel_height,
                output_array,
                stride, bias):
    """
    calculate convolution for 2D 3D
    """
    channel_number = input_array.ndim
    output_width = output_array.shape[2]
    output_height = output_array.shape[1]
    input_transform = np.zeros((output_height * output_width, kernel_array.shape[0]))
    for i in range(output_height):
        for j in range(output_width):
            input_transform[j + output_height * i, :] = get_patch(input_array, i, j, kernel_width, kernel_height,
                                                                  stride).reshape(kernel_array.shape[0])
    output_array_transform = np.dot(input_transform, kernel_array)

    return output_array_transform.reshape(output_array.shape)


def conv_matrix_ex(input_array,
                   kernel_array,
                   kernel_width,
                   kernel_height,
                   output_array,
                   stride, bias):
    """
    calculate convolution for 2D 3D
    """
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    input_transform = np.zeros((output_height * output_width, kernel_array.shape[0]))
    for i in range(output_height):
        for j in range(output_width):
            input_transform[j + output_height * i, :] = get_patch(input_array, i, j, kernel_width, kernel_height,
                                                                  stride).reshape(kernel_array.shape[0])
    output_array_transform = np.dot(input_transform, kernel_array)

    return output_array_transform.reshape(output_array.shape)
