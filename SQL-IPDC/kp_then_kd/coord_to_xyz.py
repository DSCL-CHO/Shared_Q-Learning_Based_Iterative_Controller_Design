# from gridworld import GridWorld 
def coord_to_xyz(coord, grid_shape=(4, 4), origin=(0.3, -0.1, 0.2), cell_size=0.1):
# def coord_to_xyz(coord, env, origin=(0.3, -0.1, 0.2), cell_size=0.1):
    row, col = coord
    # rows, cols = env.shape
    # rows, cols = grid_shape
    x = origin[0] + col * cell_size
    y = origin[1] - row * cell_size  
    z = origin[2]                 
    return [x, y, z]
