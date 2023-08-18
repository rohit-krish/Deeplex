from graphviz import Digraph

def _truncate_string(s, max_length):
    if len(s) > max_length:
        return s[: max_length - 3] + "..."  # truncate and add ellipsis
    return s


def _trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_graph(root, max_data_length=50):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    nodes, edges = _trace(root)
    for n in nodes:
        uid = str(id(n))
        label_data = _truncate_string(repr(n.data), max_data_length)
        label_grad = _truncate_string(repr(n.grad), max_data_length)
        dot.node(
            name=uid,
            label="{ %s | data %s | grad %s }" % (n.label, label_data, label_grad),
            shape="record",
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def _broadcast_axis(shape_left, shape_right):
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
