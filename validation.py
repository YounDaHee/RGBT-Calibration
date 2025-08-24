import cv2
import numpy as np
import util
import random
import glob
import os
import matplotlib.pyplot as plt

# 저장된 reprojection error 출력
def reprojection_error(thermal_file, rgb_file, extrin_file) :
    thermal_data = np.load(thermal_file, allow_pickle=True)
    rgb_data = np.load(rgb_file, allow_pickle=True)
    extrin_data = np.load(extrin_file, allow_pickle=True)

    print(f"Thermal Reprojection Error : {thermal_data['reproj_error']}" )
    print(f"RGB Reprojection Error : {rgb_data['reproj_error']}" )
    print(f"Extrinsic Calibration Reprojection Error : {extrin_data['reproj_error']}" )
    print(f"Rotation Matirx Determinant : {np.linalg.det(extrin_data['R'])}")

def get_all_matrix(extrin_file) :
    extrin_data = np.load(extrin_file, allow_pickle=True)

    print("[Thermal Camera-Matrix]")
    print(extrin_data['mtx1'])
    print("[Thermal Distortion-Vector]")
    print(extrin_data['dist1'])
    print("\n[RGB Camera-Matrix]")
    print(extrin_data['mtx2'])
    print("[RGB Distortion-Vector]")
    print(extrin_data['dist2'])
    print("\n[Extrinsic Matrix]")
    R = extrin_data['R']
    T = extrin_data['T']
    print(np.concatenate((R, T), axis = 1))

def get_reprojection_img(objpoints, imgpoints, rvecs, tvecs, mtx, dist) :
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    return mean_error/len(objpoints)

def rectified_images(thermal_file, rgb_file, extrin_file, thermal_folder, rgb_folder):
    # Load npz data
    thermal_data = np.load(thermal_file, allow_pickle=True)
    rgb_data = np.load(rgb_file, allow_pickle=True)
    extrin_data = np.load(extrin_file, allow_pickle=True)

    is_vertical = abs(extrin_data['T'][1, -1])>abs(extrin_data['T'][0,-1])

    keys = [k.split("_")[-1] for k in list(rgb_data['corner_storage'].item().keys())]

    #for key in keys[0] :
    key = random.sample(keys, k=min(1, len(keys)))[0]
    print(f"[Rectified image] thermal_{key} rgb_{key}")
    
    img_dim = thermal_data['img_size']#(img_w, img_h)  # (width, height)

    rgb_w, rgb_h = rgb_data['img_size']
    th_w, th_h = thermal_data['img_size']

    # Scale factors for RGB camera matrix
    if(th_w!=rgb_w) :
        scale_x = th_w / rgb_w
        scale_y = th_h / rgb_h

    # Scale the RGB camera matrix to match resized image
    rgb_cam_mtx = rgb_data['camera_matrix'].copy()
    if(th_w!=rgb_w) :
        rgb_cam_mtx[0, 0] *= scale_x  # fx
        rgb_cam_mtx[0, 2] *= scale_x  # cx
        rgb_cam_mtx[1, 1] *= scale_y  # fy
        rgb_cam_mtx[1, 2] *= scale_y  # cy

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        thermal_data['camera_matrix'], thermal_data['distortion_coeffs'],
        rgb_cam_mtx, rgb_data['distortion_coeffs'],
        img_dim, extrin_data['R'], extrin_data['T'],
        #flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.7
    )

    P1 = P1[:3, :3]
    P2 = P2[:3, :3]

    # Undistort + rectify maps
    mapx_l, mapy_l = cv2.initUndistortRectifyMap(
        thermal_data['camera_matrix'], thermal_data['distortion_coeffs'],
        R1, P1, img_dim, cv2.CV_32FC1
    )

    mapx_r, mapy_r = cv2.initUndistortRectifyMap(
        rgb_cam_mtx, rgb_data['distortion_coeffs'],
        R2, P2, img_dim, cv2.CV_32FC1
    )

    # Load images
    thermal_img = util.preprocessing(
            cv2.imread(f"{thermal_folder}/thermal_{key}", cv2.IMREAD_GRAYSCALE))
    rgb_img = cv2.imread(f"{rgb_folder}/rgb_{key}", cv2.IMREAD_GRAYSCALE)

    if(th_w!=rgb_w):
        rgb_img = cv2.resize(rgb_img, img_dim)

    l_coner = thermal_data['corner_storage'].item()[f'thermal_{key}'].copy()
    r_coner = rgb_data['corner_storage'].item()[f'rgb_{key}'].copy()
    if(th_w!=rgb_w) :
            r_coner[...,0] *= scale_x
            r_coner[...,1] *= scale_y

    thermal_img = util.draw_points_with_index(thermal_img, l_coner)
    rgb_img = util.draw_points_with_index(rgb_img,r_coner)

    # before rectification
    util.draw_epipolar_lines(thermal_img, rgb_img, l_coner.reshape(-1,2), is_vertical, 'before_combine.png')

    # Apply remapping
    limg_rect = cv2.remap(thermal_img, mapx_l, mapy_l, interpolation=cv2.INTER_LINEAR)
    rimg_rect = cv2.remap(rgb_img, mapx_r, mapy_r, interpolation=cv2.INTER_LINEAR)

    L_u = cv2.undistortPoints(l_coner.reshape(-1,1,2), 
                thermal_data['camera_matrix'], thermal_data['distortion_coeffs'], 
                R=R1, P=P1).reshape(-1,2)

    # after rectification
    util.draw_epipolar_lines(limg_rect, rimg_rect, L_u, is_vertical, 'combine.png')

    rms = []
    for k in keys:
        l_coner = thermal_data['corner_storage'].item()[f'thermal_{k}'].copy()
        r_coner = rgb_data['corner_storage'].item()[f'rgb_{k}'].copy()
        if(th_w!=rgb_w) :
            r_coner[...,0] *= scale_x
            r_coner[...,1] *= scale_y
        L_u = cv2.undistortPoints(l_coner.reshape(-1,1,2), 
                              thermal_data['camera_matrix'], thermal_data['distortion_coeffs'], 
                              R=R1, P=P1).reshape(-1,2)
        R_u = cv2.undistortPoints(r_coner.reshape(-1,1,2), 
                                rgb_cam_mtx, rgb_data['distortion_coeffs'], 
                                R=R2, P=P2).reshape(-1,2)
        if is_vertical :
            rms.append(float(np.sqrt(((L_u[:,0]-R_u[:,0])**2).mean())))
        else :
            rms.append(float(np.sqrt(((L_u[:,1]-R_u[:,1])**2).mean())))
        
    if is_vertical : 
        print(f"rectified x-RMS: {np.array(rms).mean():.3f} px")
    else :
        print(f"rectified y-RMS: {np.array(rms).mean():.3f} px")

# intrinsic matrix가 과적합 되지 않았는지 여부 확인(다른 이미지 셋에 대해서 camera matrix 적용)
def validation_with_other_img(val_file, test_file, interval = 100.0, pattern_size = (4, 6)):
    # Load npz data
    val_data = np.load(val_file, allow_pickle=True)
    test_data = np.load(test_file, allow_pickle=True)

    imgpoints = list(test_data['corner_storage'].item().values())
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= interval

    objpoints = np.array([objp for i in range(len(imgpoints))])

    imgpoints = np.array(imgpoints)

    rvecs = []
    tvecs = []
    for i in range(len(objpoints)) :
        _,rvec,tvec = cv2.solvePnP(objpoints[i], imgpoints[i], val_data['camera_matrix'], 
                                    val_data['distortion_coeffs'], flags=cv2.SOLVEPNP_ITERATIVE)
        rvecs.append(rvec)
        tvecs.append(tvec)

    result = util.reprojection_error(objpoints, imgpoints, rvecs, tvecs 
                            , val_data['camera_matrix'], val_data['distortion_coeffs'])
    
    print(result)

# 축별 reprojection error 확인 -> 어느 축에서 에러가 크게 감지되는지 확인
def residual_axis_RMSE(val_file, test_file, interval = 100.0, pattern_size = (4, 6)):
    # Load npz data
    val_data = np.load(val_file, allow_pickle=True)
    test_data = np.load(test_file, allow_pickle=True)

    imgpoints = list(test_data['corner_storage'].item().values())
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= interval

    objpoints = np.array([objp for i in range(len(imgpoints))])

    imgpoints = np.array(imgpoints)

    rvecs = []
    tvecs = []
    for i in range(len(objpoints)) :
        ok,rvec,tvec = cv2.solvePnP(objpoints[i], imgpoints[i], val_data['camera_matrix'], 
                                    val_data['distortion_coeffs'], flags=cv2.SOLVEPNP_ITERATIVE)
        rvecs.append(rvec)
        tvecs.append(tvec)

    x_mean_error = 0
    y_mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], val_data['camera_matrix'], val_data['distortion_coeffs'])
        x_error = (cv2.norm(imgpoints[i,:,:,0], imgpoints2[:,:,0], cv2.NORM_L2)/len(imgpoints2))**0.5
        y_error = (cv2.norm(imgpoints[i,:,:,1], imgpoints2[:,:,1], cv2.NORM_L2)/len(imgpoints2))**0.5
        x_mean_error += x_error
        y_mean_error += y_error
    print(f'x_mean_error : {x_mean_error/len(objpoints)}')
    print(f'y_mean_error : {y_mean_error/len(objpoints)}') 
    
def estimate_Homograpy(thermal_file, rgb_file, thermal_folder, rgb_folder) :
    thermal_data = np.load(thermal_file, allow_pickle=True)
    rgb_data = np.load(rgb_file, allow_pickle=True)

    thermal_coner = thermal_data['corner_storage'].item()
    rgb_coner = rgb_data['corner_storage'].item()

    pts1 = []
    pts2 = []
    for k in rgb_coner :
        th_undist_corners = cv2.undistortPoints(thermal_coner[f'thermal_{k.split("_")[-1]}'],
                                                thermal_data['camera_matrix'],
                                                thermal_data['distortion_coeffs'],
                                                P = thermal_data['camera_matrix'])
        pts1.append(th_undist_corners)
        rgb_undist_corners = cv2.undistortPoints(rgb_coner[k], 
                                                 rgb_data['camera_matrix'],
                                                 rgb_data['distortion_coeffs'],
                                                 P = rgb_data['camera_matrix'])
        pts2.append(rgb_undist_corners)

    # 대응점 (Nx2)
    pts1 = np.array(pts1, dtype=np.float32).reshape(-1, 2) # RGB 영상
    pts2 = np.array(pts2, dtype=np.float32).reshape(-1, 2)  # IR 영상

    # RANSAC 기반 Homography 추정
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 1.0)
    
    output_dir = 'image_registration'
    os.makedirs(output_dir, exist_ok=True)
    proved_image = glob.glob(f'{thermal_folder}/*.png')
    for fname in proved_image :
        th_img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        th_img = util.preprocessing(th_img)
        th_img = cv2.undistort(th_img, thermal_data['camera_matrix'],
                                                thermal_data['distortion_coeffs'])
        rgb_img = cv2.imread(f'{rgb_folder}/rgb_{fname.split("_")[-1]}', cv2.IMREAD_GRAYSCALE)
        rgb_img = cv2.undistort(rgb_img, rgb_data['camera_matrix'],
                                                 rgb_data['distortion_coeffs'])
        
        h, w = rgb_img.shape[:2]
        aligned_ir = cv2.warpPerspective(th_img, H, (w,h))

        aligned_ir = rgb_img/2 + aligned_ir

        cv2.imwrite(f"{output_dir}/{fname.split('_')[-1]}", aligned_ir)


def triangulation(train_folder, val_folder):
    # Load npz data
    extrin_data = np.load(f'{train_folder}/thermal_rgb_extrinsics.npz', allow_pickle=True)

    th_w, th_h = extrin_data['img_size1']
    rgb_w, rgb_h = extrin_data['img_size2']
    
    # Scale factors for RGB camera matrix
    if(th_w!=rgb_w) :
        scale_x = th_w / rgb_w
        scale_y = th_h / rgb_h

    # Scale the RGB camera matrix to match resized image
    rgb_cam_mtx = extrin_data['mtx2'].copy()
    if(th_w!=rgb_w) :
        rgb_cam_mtx[0, 0] *= scale_x  # fx
        rgb_cam_mtx[0, 2] *= scale_x  # cx
        rgb_cam_mtx[1, 1] *= scale_y  # fy
        rgb_cam_mtx[1, 2] *= scale_y  # cy

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        extrin_data['mtx1'], extrin_data['dist1'],
        rgb_cam_mtx, extrin_data['dist2'],
        extrin_data['img_size1'], extrin_data['R'], extrin_data['T'],
        #flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.7
    )

    thermal_data = np.load(f'{val_folder}/calibration_data_with_centers_thermal.npz', allow_pickle=True)
    rgb_data = np.load(f'{val_folder}/calibration_data_with_centers_rgb.npz', allow_pickle=True)

    #for k in extrin_data['matched_ids'][0]:
    k = list(rgb_data['corner_storage'].item().keys())
    k = k[20].split("_")[-1]
    print(k)
    l_coner = thermal_data['corner_storage'].item()[f'thermal_{k}'].copy()
    r_coner = rgb_data['corner_storage'].item()[f'rgb_{k}'].copy()
    if(th_w!=rgb_w) :
        r_coner[...,0] *= scale_x
        r_coner[...,1] *= scale_y
    pt1 = cv2.undistortPoints(l_coner.reshape(-1,1,2), 
                            extrin_data['mtx1'], extrin_data['dist1'], 
                            R=R1, P=P1).reshape(-1,2)
    pt2 = cv2.undistortPoints(r_coner.reshape(-1,1,2), 
                            rgb_cam_mtx, extrin_data['dist2'], 
                            R=R2, P=P2).reshape(-1,2)
   
    d3_point = cv2.triangulatePoints(P1, P2, pt1.T, pt2.T)
    d3_point = (d3_point[:3,:]/d3_point[3:4,:]).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(d3_point[:,0], d3_point[:,1], d3_point[:,2], c='b', marker='o')
    ax.view_init(elev=60, azim=45)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

     
