import numpy as np 
import cupy as cp
import cupyx as cpx
import time
import cv2

from gpu_acc_utils import *
from cupyx.scipy.interpolate import RegularGridInterpolator as reg_interp

def reshape_grids_gpu(grid_x, grid_y, grid_z):
    x_vec = grid_x.T.reshape(-1, 1)
    y_vec = grid_y.T.reshape(-1, 1)
    z_vec = grid_z.T.reshape(-1, 1)

    xyz = cp.concatenate([x_vec, y_vec, z_vec], axis=1)

    return xyz

def CIRNangles2R_gpu(azimuth, tilt, swing):
    R = np.empty((3,3))

    R[0,0] = -np.cos(azimuth) * np.cos(swing) - np.sin(azimuth) * np.cos(tilt) * np.sin(swing)
    R[0,1] = np.cos(swing) * np.sin(azimuth) - np.sin(swing) * np.cos(tilt) * np.cos(azimuth)
    R[0,2] = -np.sin(swing) * np.sin(tilt)
    R[1,0] = -np.sin(swing) * np.cos(azimuth) + np.cos(swing) * np.cos(tilt) * np.sin(azimuth)
    R[1,1] = np.sin(swing) * np.sin(azimuth) + np.cos(swing) * np.cos(tilt) * np.cos(azimuth)
    R[1,2] = np.cos(swing) * np.sin(tilt);
    R[2,0] = np.sin(tilt) * np.sin(azimuth)
    R[2,1] = np.sin(tilt) * np.cos(azimuth)
    R[2,2] = -np.cos(tilt)
    
    R_gpu = cp.array(R)

    return R_gpu

def intrinsicsExtrinsics2P_gpu(intrinsics, extrinsics):
    K = cp.zeros((3,3))
    K[0,0] = -intrinsics[4]
    K[1,1] = -intrinsics[5]
    K[0,2] = intrinsics[2]
    K[1,2] = intrinsics[3]
    K[2,2] = 1

    azimuth = extrinsics[3]
    tilt = extrinsics[4]
    swing = extrinsics[5]
    R = CIRNangles2R_gpu(azimuth, tilt, swing)

    x = extrinsics[0]
    y = extrinsics[1]
    z = extrinsics[2]
    column_vec = cp.array([-x, -y, -z]).reshape(-1, 1)
    IC = cp.concatenate([cp.eye(3), column_vec], axis=1)

    P = cp.dot(K, cp.dot(R, IC))
    P /= P[2, 3]

    return P, K, R, IC

def distortUV_gpu(UV, intrinsics):
    NU = intrinsics[0]
    NV = intrinsics[1]
    c0U = intrinsics[2]
    c0V = intrinsics[3]
    fx = intrinsics[4]
    fy = intrinsics[5]
    d1 = intrinsics[6]
    d2 = intrinsics[7]
    d3 = intrinsics[8]
    t1 = intrinsics[9]
    t2 = intrinsics[10]

    U = UV[0, :]
    V = UV[1, :]

    x = (U - c0U) / fx
    y = (V - c0V) / fy

    # Radial distortion
    r2 = x**2 + y**2
    fr = 1 + d1 * r2 + d2 * r2**2 + d3 * r2**3

    # Tangential distortion
    dx = 2 * t1 * x * y + t2 * (r2 + 2 * x**2)
    dy = t1 * (r2 + 2 * y**2) + 2 * t2 * x * y

    # Apply correction
    xd = x * fr + dx
    yd = y * fr + dy
    Ud = xd * fx + c0U
    Vd = yd * fy + c0V

    # Find negative UV coordinates
    flag_mask = (Ud < 0) | (Ud > NU) | (Vd < 0) | (Vd > NV)
    Ud[flag_mask] = 0
    Vd[flag_mask] = 0

    # Define corners of the image
    Um = cp.array([0, 0, NU.item(), NU.item()])
    Vm = cp.array([0, NV.item(), NV.item(), 0])

    # Normalization
    xm = (Um - c0U) / fx
    ym = (Vm - c0V) / fy
    r2m = xm**2 + ym**2

    # Tangential Distortion at corners
    dxm = 2 * t1 * xm * ym + t2 * (r2m + 2 * xm**2)
    dym = t1 * (r2m + 2 * ym**2) + 2 * t2 * xm * ym

    # Find values larger than those at corners
    max_dym = cp.max(cp.abs(dym))
    max_dxm = cp.max(cp.abs(dxm))

    # Indices where distortion values are larger than those at corners
    exceeds_dy = cp.where(cp.abs(dy) > max_dym)
    exceeds_dx = cp.where(cp.abs(dx) > max_dxm)

    # Initialize flag array (assuming it’s previously defined)
    flag = cp.ones_like(Ud)
    flag[exceeds_dy] = 0.0
    flag[exceeds_dx] = 0.0

    return Ud, Vd, flag

def xyz2DistUV_gpu(intrinsics, extrinsics, grid_x, grid_y, grid_z):
    P_gpu, K_gpu, R_gpu, IC_gpu = intrinsicsExtrinsics2P_gpu(intrinsics, extrinsics)

    xyz = reshape_grids_gpu(grid_x, grid_y, grid_z)
    xyz_homogeneous = cp.vstack((xyz.T, cp.ones(xyz.shape[0])))
    
    UV_homogeneous = cp.dot(P_gpu, xyz_homogeneous)
    UV = UV_homogeneous[:2, :] / UV_homogeneous[2, :]

    Ud, Vd, flag = distortUV_gpu(UV, intrinsics)

    DU = Ud.reshape(grid_x.shape, order="F")
    DV = Vd.reshape(grid_y.shape, order="F")
    
    # Compute camera coordinates
    xyzC = cp.dot(cp.dot(R_gpu, IC_gpu), xyz_homogeneous)

    # Find negative Zc coordinates (Z <= 0) and update the flag
    negative_z_indices = cp.where(xyzC[2, :] <= 0.0)
    flag[negative_z_indices] = 0.0
    flag = flag.reshape(grid_x.shape, order="F")

    return DU * flag, DV * flag

def getPixels_gpu(image, Ud, Vd, s):

    """
    Pulls rgb or gray pixel intensities from image at specified
    pixel locations corresponding to X,Y coordinates calculated in either
    xyz2DistUV or dlt2UV.

    Args:
        image (ndarray): image where pixels will be taken from
        Ud: Nx1 vector of distorted U coordinates for N points
        Vd: Nx1 vector of distorted V coordinates for N points
        s: shape of output image

    Returns:
        ir (ndarray): pixel intensities

    """

    # Use regular grid interpolator to grab points
    im_s = image.shape
    if len(im_s) > 2:
        ir = cp.full((s[0], s[1], im_s[2]), cp.nan)
        for i in range(im_s[2]):
            rgi = reg_interp(
                (cp.arange(0, image.shape[0]), cp.arange(0, image.shape[1])),
                image[:, :, i],
                bounds_error=False,
                fill_value=cp.nan,
            )
            ir[:, :, i] = rgi((Vd, Ud))
    else:
        ir = cp.full((s[0], s[1], 1), cp.nan)
        rgi = reg_interp(
            (cp.arange(0, image.shape[0]), cp.arange(0, image.shape[1])),
            image,
            bounds_error=False,
            fill_value=np.nan,
        )
        ir[:, :, 0] = rgi((Vd, Ud))

    # # Mask out values out of range
    
    mask_u = cp.logical_or(Ud <= 1, Ud >= image.shape[1])
    mask_v = cp.logical_or(Vd <= 1, Vd >= image.shape[0])
    mask = cp.logical_or(mask_u, mask_v)

    # Use cp.where to assign NaN where the mask is True
    if len(im_s) > 2:
        mask = mask[:, :, None]  # Adding a channel dimension (matching ir's shape)
        ir = cp.where(mask, cp.nan, ir)  # For multi-channel data
    else:
        ir[mask] = cp.nan # For 2D data

    return ir

def mergeRectify_gpu(image_path, intrinsics, extrinsics, grid_x, grid_y, grid_z):
    
    s = grid_x.shape

    I = cv2.imread(image_path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    
    # Convert the NumPy array to a CuPy array
    I_gpu = cp.asarray(I)

    Ud, Vd = xyz2DistUV_gpu(intrinsics, extrinsics, grid_x, grid_y, grid_z)

    Ud = cp.round(Ud)
    Vd = cp.round(Vd)
    Ud = Ud.astype(int)
    Vd = Vd.astype(int)
    
    ir = getPixels_gpu(I_gpu, Ud, Vd, s)
    ir = cp.array(ir, dtype=np.uint8)

    return ir

def mergeRectifyFolder_gpu(folder_path, intrinsics, extrinsics, grid_x, grid_y, grid_z, zarr_store_path):
    
    s = grid_x.shape

    Ud, Vd = xyz2DistUV_gpu(intrinsics, extrinsics, grid_x, grid_y, grid_z)
    Ud = cp.round(Ud).astype(int)
    Vd = cp.round(Vd).astype(int)

    # Open the Zarr store once before the loop
    store = zarr.open_group(zarr_store_path, mode='a')
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        I = cv2.imread(image_path)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        
        I_gpu = cp.asarray(I)
        
        ir = getPixels_gpu(I_gpu, Ud, Vd, s)
        ir = cp.array(ir, dtype=np.uint8)
        
        # Create a dataset name by appending 'rectified' to the original image name
        dataset_name = f"{os.path.splitext(image_name)[0]}_rectified"
        
        # Save the ir array to the Zarr store
        # store.create_dataset(dataset_name, data=ir, compression='zlib')
        store[dataset_name] = ir.get()
    
    return store

def mergeRectifyLabelsFolder_gpu(folder_path, intrinsics, extrinsics, grid_x, grid_y, grid_z, zarr_store_path):
    
    s = grid_x.shape

    Ud, Vd = xyz2DistUV_gpu(intrinsics, extrinsics, grid_x, grid_y, grid_z)
    Ud = cp.round(Ud).astype(int)
    Vd = cp.round(Vd).astype(int)

    # Open the Zarr store once before the loop
    store = zarr.open_group(zarr_store_path, mode='a')
    
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        
        I_gpu = cp.asarray(I)
        
        ir = getPixels_gpu(I_gpu, Ud, Vd, s)
        ir = cp.array(ir, dtype=np.uint8)
        
        # Create a dataset name by appending 'rectified' to the original image name
        dataset_name = f"{os.path.splitext(image_name)[0]}_rectified"
        
        # Save the ir array to the Zarr store
        # store.create_dataset(dataset_name, data=ir, compression='zlib')
        store[dataset_name] = ir.get()
    
    return store