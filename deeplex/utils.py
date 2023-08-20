def broadcast_axis__(shape_left, shape_right):
    """
    Determine the axes along which broadcasting occurs between two tensor shapes.

    This function identifies the axes along which broadcasting will happen when two tensors
    with different shapes are used in operations. Broadcasting allows compatible tensors
    to be combined element-wise even when their shapes are not exactly the same.

    Args:
        shape_left (tuple): The shape of the left tensor.
        shape_right (tuple): The shape of the right tensor.

    Returns:
        tuple: A tuple containing two inner tuples representing the axes along which
               broadcasting occurs for the left and right tensors, respectively. If no
               broadcasting is necessary, both inner tuples are empty.

    Example:
        >>> shape_left = (3, 1, 5)
        >>> shape_right = (1, 5)
        >>> broadcast_axis__(shape_left, shape_right) # ((0, 1), (0,))
        In this example, the first and second axes of the left tensor will be broadcasted
        to match the corresponding dimensions of the right tensor along the first axis.
    """

    if shape_left == shape_right:
        return ((), ())

    left_dim = len(shape_left)
    right_dim = len(shape_right)
    result_ndim = max(left_dim, right_dim)

    left_padded = (1,) * (result_ndim - left_dim) + shape_left
    right_padded = (1,) * (result_ndim - right_dim) + shape_right

    left_axes = []
    right_axes = []

    for axis_idx, (left_axis, right_axis) in enumerate(zip(left_padded, right_padded)):
        if right_axis > left_axis:
            left_axes.append(axis_idx)
        elif left_axis > right_axis:
            right_axes.append(axis_idx)

    return tuple(left_axes), tuple(right_axes)
