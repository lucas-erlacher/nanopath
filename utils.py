def assert_shape(tensor, expected_shape, tensor_name):
    actual_shape = tuple(tensor.shape)
    if actual_shape != tuple(expected_shape):
        raise RuntimeError(f"{tensor_name} shape {actual_shape} diverges from expected {tuple(expected_shape)}")