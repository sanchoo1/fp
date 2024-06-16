import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_coordinate_system(ax, T, label, length=50):
    """
    Draw a coordinate system given a transformation matrix.
    
    Args:
    ax (Axes3D): The 3D axis to draw on.
    T (np.array): The 4x4 transformation matrix.
    label (str): Label for the coordinate system.
    length (float): Length of the coordinate system axes.
    """
    # Origin of the coordinate system
    origin = T[:3, 3]

    # Axes of the coordinate system
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    ax.quiver(*origin, *x_axis, color='r', length=length)
    ax.quiver(*origin, *y_axis, color='g', length=length)
    ax.quiver(*origin, *z_axis, color='b', length=length)

    ax.text(*origin, label, fontsize=12, fontweight='bold')

def plot_marker_and_camera(marker_T):
    """
    Plot the marker and camera coordinate systems in 3D space.
    
    Args:
    marker_T (np.array): The 4x4 transformation matrix of the marker.
    camera_T (np.array): The 4x4 transformation matrix of the camera.
    """

    camera_T = np.eye(4)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw the marker coordinate system
    draw_coordinate_system(ax, camera_T, 'Camera', length=300)

    # Transform camera_T to marker coordinate system
    T_cm = np.dot(np.linalg.inv(camera_T), marker_T)

    # Draw the camera coordinate system
    draw_coordinate_system(ax, T_cm, 'Marker', length=300)

    # Set plot limits
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
    ax.set_zlim([-400, 2300])

    # Set labels with units
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    plt.show()



markre_T_new = np.array([
[  0,   0.7071,    0.7071 ,     0],
[  1 ,    0,  0   ,  -110         ],
[   0,  0.7071 ,   -0.7071  ,   2000],
[0,0,0,1]

])

plot_marker_and_camera(markre_T_new)

# camera_T_new = np.array([
#     [0.99572, 0.064361, 0.066303, 131.973],
#     [0.063485, -0.99787, 0.015232, -15.241],
#     [0.067142, -0.010957, -0.99768, 538.46],
#     [0, 0, 0, 1]
# ])

# plot_marker_and_camera(camera_T_new)