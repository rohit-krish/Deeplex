def broadcast_axis__(shape_left, shape_right):
    """
    Determine the axes along which broadcasting occurs between two shapes.

    Args:
        shape_left: Shape of the left tensor.
        shape_right: Shape of the right tensor.

    Returns:
        A tuple of two tuples representing the axes along which broadcasting occurs.
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
