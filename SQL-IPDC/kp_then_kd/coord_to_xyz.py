# from gridworld import GridWorld 
# (row, col) → EE 3D 위치 매핑 함수 

def coord_to_xyz(coord, grid_shape=(4, 4), origin=(0.3, -0.1, 0.2), cell_size=0.1):
# def coord_to_xyz(coord, env, origin=(0.3, -0.1, 0.2), cell_size=0.1):
    """
    - origin: 그리드의 좌상단이 매핑되는 실제 위치
    - cell_size: 각 셀 간 거리 (단위: m)
    """
    row, col = coord
    # rows, cols = env.shape
    # rows, cols = grid_shape
    
    x = origin[0] + col * cell_size
    y = origin[1] - row * cell_size  # row가 아래로 내려갈수록 y 감소
    z = origin[2]                    # 고정된 높이


    return [x, y, z]