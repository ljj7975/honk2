def num_floats_to_GB(num_floats):
    # float is 4 bytes
    # 1e+9 bytes ~ 1 gigabytes
    return round((num_floats * 4) / 1e+9, 3)
