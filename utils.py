import numpy as np

def create_circular_mask(shape, voxel_spacing=(1.0,1.0), center=None, radius=None):
    # Create a cirucular mask for  height h, width w and depth d centered at center with given radius

    #extract depth, width and height
#     print(shape)
    w,h = shape

    if center is None:  # use the middle of the image
        center = np.array(np.array(shape)/2,dtype=int)

    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center, shape-center)

    Y,X=(np.ogrid[:w, :h])
    dist_from_center = ((X - center[0]) * voxel_spacing[0] )** 2  +\
                       ((Y - center[1]) * voxel_spacing[1]) ** 2

    mask = dist_from_center <= radius ** 2
    return mask

def create_spherical_mask(shape, voxel_spacing=(1.0,1.0,1.0), center=None, radius=None):
    # Create a spherical mask for  height h, width w and depth d centered at center with given radius

    #extract depth, width and height
    d,w,h = shape

    if center is None:  # use the middle of the image
        center = np.array(np.array(shape)/2,dtype=int)

    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center, shape-center)

    Z,X,Y=(np.ogrid[:d, :w, :h])
    dist_from_center = ((Z - center[0])*voxel_spacing[0])**2      +\
                       ((X - center[1]) * voxel_spacing[1]) ** 2  +\
                       ((Y - center[2]) * voxel_spacing[2]) ** 2

    mask = dist_from_center <= radius ** 2
    return mask

