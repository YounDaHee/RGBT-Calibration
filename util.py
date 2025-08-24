import cv2
import numpy as np
import random

# 이미지의 대비 조절
def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def preprocessing(img) :
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(1, 1))
    img = clahe.apply(img)
    img = gamma_correction(img, gamma=1.6)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=75)
    return img    

# 캘리브레이션 시, 노이즈에 의해 grid_point가 기준점에서 너무 벋어났는지 여부 확인
# 기울기 tolerence = 0.2, -20 degree ~ + 20 degree 오차 허용
def legal_calibration(pattern_size, center, t = 0.2, e = 0.1e-05) :
    center = center.reshape(24, 2)
    for col in range(0, pattern_size[1]) :
        diff = None
        for row in range(1, pattern_size[0]) :
            diff_x = center[row+col*pattern_size[0]][0] - center[row+col*pattern_size[0]-1][0]
            diff_y = center[row+col*pattern_size[0]][1] - center[row+col*pattern_size[0]-1][1]
            if(diff is not None) :
                if ((diff < (diff_x/(diff_y) - t)) or (diff > (diff_x/(diff_y) + t))) :
                    return False
            diff = diff_x/diff_y

    # for row in range(0, pattern_size[0]) :
    #     diff = None
    #     for col in range(1, pattern_size[1]) :
    #         diff_x = center[row+col*pattern_size[0]][0] - center[row+(col-1)*pattern_size[0]][0]
    #         diff_y = center[row+col*pattern_size[0]][1] - center[row+(col-1)*pattern_size[0]][1]
    #         if(diff is not None) :
    #             if(diff != diff_x//diff_y) :
    #                 print(f'{row+col*pattern_size[0]}, {row+(col-1)*pattern_size[0]}')
    #                 print(row)
    #                 return False
    #         diff = diff_x//diff_y

    return True

# 동일하게 감지하는 이미지 셋을 찾고 그 중 50개를 랜덤으로 선택
def common_image(A, B, k = 50) :
    a_img = {fname.split("_")[-1] for fname in A.center_storage.keys()}
    b_img = {fname.split("_")[-1] for fname in B.center_storage.keys()}
    
    common_all = list(a_img & b_img)
    assert len(common_all)>0, "[ERROR] No Common Image"
    if k>0 :
        return random.sample(common_all, k=min(k, len(common_all)))
    else : 
        return common_all

def remove_flip(pattern_size, center) :
    rows, cols = pattern_size[1], pattern_size[0]
    centers_2d = center.reshape(rows, cols, 2)

    # 왼쪽 위와 오른쪽 위 점의 x좌표 비교 → 좌우 뒤집힘 판별
    if centers_2d[0, 0, 1] > centers_2d[0, -1, 1]:
        centers_2d = centers_2d[:, ::-1, :]

    # 왼쪽 위와 왼쪽 아래 점의 y좌표 비교 → 상하 뒤집힘 판별
    if centers_2d[0, 0, 0] > centers_2d[-1, 0, 0]:
        centers_2d = centers_2d[::-1, :, :]

    center = centers_2d.reshape(-1, 1, 2)

    return center


def draw_points_with_index(img, centers):
    debug_img = img.copy()

    # centers를 int로 변환
    for idx, pt in enumerate(centers):
        x, y = int(pt[0][0]), int(pt[0][1])
        cv2.circle(debug_img, (x, y), 5, (0, 0, 255), -1)  # 빨간 점
        cv2.putText(debug_img, str(idx), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return debug_img

def reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist) :
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = (cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2))
        mean_error += error
    return mean_error/len(objpoints)

# 에피폴라 선 그려줌
def draw_epipolar_lines(limg_rect, rimg_rect, coners, is_vertical, name):
    """
    두 rectified 이미지를 나란히 붙이고, 일정 간격으로 에피폴라 라인을 그림.
    """
    # 색상을 BGR로 변환 (라인 색을 넣기 위해)
    if len(limg_rect.shape) == 2:
        limg_color = cv2.cvtColor(limg_rect, cv2.COLOR_GRAY2BGR)
    else:
        limg_color = limg_rect.copy()

    if len(rimg_rect.shape) == 2:
        rimg_color = cv2.cvtColor(rimg_rect, cv2.COLOR_GRAY2BGR)
    else:
        rimg_color = rimg_rect.copy()

    # 두 이미지를 가로로 붙임
    if is_vertical :
        combined = np.vstack((limg_color, rimg_color))
    else :
        combined = np.vstack((limg_color, rimg_color))
    h, w, _ = combined.shape

    # 라인 색상 목록
    colors = [
        (0, 0, 255),    # 빨강
        (0, 255, 0),    # 초록
        (255, 0, 0),    # 파랑
        (0, 255, 255),  # 노랑
        (255, 0, 255),  # 보라
        (255, 255, 0)   # 하늘
    ]

    # 일정 간격으로 y좌표 선택
    if is_vertical :
        for idx, x in enumerate(coners[:, 0]):
            color = colors[idx % 4]
            cv2.line(combined, (int(x), 0), (int(x), h), color, 1)
    else :
        for idx, y in enumerate(coners[:, 1]):
            color = colors[idx // 4]
            cv2.line(combined, (int(y), 0), (int(y), w), color, 1)

    cv2.imwrite(name, combined)