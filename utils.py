def assert_shape(tensor, expected_shape, tensor_name):
    actual_shape = tuple(tensor.shape)
    expected_shape = tuple(expected_shape)

    if len(actual_shape) != len(expected_shape):
        raise RuntimeError(f"{tensor_name} shape {actual_shape} diverges from expected {expected_shape}")
    
    for i, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_shape)):
        if expected_dim == "*":
            continue
        
        if actual_dim != expected_dim:
            raise RuntimeError(f"{tensor_name} shape {actual_shape} diverges from expected {expected_shape} at dim {i}")