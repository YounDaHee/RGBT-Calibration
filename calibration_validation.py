import cv2
import numpy as np

def img_undistort(config, img_path):
    """왜곡 보정 수행"""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    data = np.load(config['file'], allow_pickle=True)
    K = data["camera_matrix"]
    D = data["distortion_coeffs"]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0.5)
    x, y, w, h = roi
    undistorted = cv2.undistort(img, K, D, None, new_K)
    undistorted = undistorted[y:y+h, x:x+w]
    
    return undistorted

def undistort(img, K, D, R_rect, K_common, target_size):
    """
    원본 카메라(K, D)에서 공통 카메라(K_common)로
    회전(R_rect) 및 왜곡보정을 한 번에 수행.
    """
    w, h = target_size
    map_x, map_y = cv2.initUndistortRectifyMap(
        K, D, R_rect, K_common, (w, h), cv2.CV_32FC1
    )
    rectified = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return rectified

def extrinsic_to_pixel_shift(K1, K2, R, t, img1, img2, Zref):
    """
    pixel 평면에서의 평균 이동량 (dx, dy) 계산
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # cam2 기준 평면 (Z=Zref)
    corners_3d_cam2 = np.array([
        [-0.050, -0.050, Zref],
        [ 0.050, -0.050, Zref],
        [ 0.050,  0.050, Zref],
        [-0.050,  0.050, Zref]
    ], dtype=np.float32)

    corners_3d_cam1 = (R @ corners_3d_cam2.T + t).T
    rvec_zero = np.zeros(3)
    img_points1, _ = cv2.projectPoints(corners_3d_cam1, rvec_zero, np.zeros(3), K1, np.zeros(5))
    img_points2, _ = cv2.projectPoints(corners_3d_cam2, rvec_zero, np.zeros(3), K2, np.zeros(5))

    img_points1 = img_points1.reshape(-1, 2)
    img_points2 = img_points2.reshape(-1, 2)

    # 정규화
    img_points1_norm = np.column_stack((img_points1[:, 0] / w1, img_points1[:, 1] / h1))
    img_points2_norm = np.column_stack((img_points2[:, 0] / w2, img_points2[:, 1] / h2))

    # 평균 shift 계산
    shift_norm = np.mean(img_points1_norm - img_points2_norm, axis=0)
    dx = shift_norm[0] * w1
    dy = shift_norm[1] * h1

    return dx, dy


def img_overlay(configs, validation):
    rgb_data = np.load(configs["rgb_incal"]["file"], allow_pickle=True)
    th_data = np.load(configs["th_incal"]["file"], allow_pickle=True)
    ex_data = np.load(configs["excal"]["file"], allow_pickle=True)

    K_rgb = np.array(ex_data["mtx1"], dtype=np.float64, copy=True)
    D_rgb = ex_data["dist1"]
    
    K_th = np.array(ex_data["mtx2"], dtype=np.float64, copy=True)
    D_th = ex_data["dist2"]

    R = ex_data["R"]
    T = ex_data["T"]

    rgb_size = tuple(rgb_data["img_size"])
    th_size = tuple(th_data["img_size"])
    
    # 이미지 읽기
    rgb_img = cv2.imread(validation["rgb_path"])
    th_img = cv2.imread(validation["th_path"])

    if rgb_size != th_size:
        rgb_img = cv2.resize(rgb_img, th_size, interpolation=cv2.INTER_LINEAR)

    # 공통 해상도는 thermal 기준으로 통일
    h, w = th_img.shape[:2]
    target_size = (w, h)

    # 공통 카메라 내장행렬 K_common 설정
    # 두 카메라 중 더 작은 focal length를 사용해서 공통 FOV를 넓게 잡음
    fx_common = min(K_th[0, 0], K_rgb[0, 0])
    fy_common = min(K_th[1, 1], K_rgb[1, 1])
    cx_common = w / 2.0
    cy_common = h / 2.0

    K_common = np.array([
        [fx_common, 0.0,       cx_common],
        [0.0,       fy_common, cy_common],
        [0.0,       0.0,       1.0]
    ])

    # (1) Undistortion + 두 이미지 intrinsic 통일(동일한 FOV를 지니도록)
    img_th_undist = undistort(th_img, K_th, D_th, np.eye(3), K_common, target_size)
    img_rgb_undist = undistort(rgb_img, K_rgb, D_rgb, np.eye(3), K_common, target_size)

    # (2) img2 회전 및 dx, dy 계산
    dx, dy = extrinsic_to_pixel_shift(K_th, K_rgb, R, T, img_th_undist, img_rgb_undist, Zref=validation["depth"])
    print(f"Estimated pixel shift (dx, dy): ({dx:.2f}, {dy:.2f})")

    # # (4) 이동 적용
    M = np.float32([[1, 0, 0], [0, 1, dy]])
    shifted = cv2.warpAffine(img_rgb_undist, M, (img_th_undist.shape[1], img_th_undist.shape[0]))

    # # # (5) 블렌딩
    blended_after = cv2.addWeighted(shifted, 0.5, img_th_undist, 0.7, 0.5)

    cv2.imwrite(f"{validation['cali_folder']}/overlay.jpg", blended_after)
    
    
