import intrinsic_calibration as incal
import extrinsic_calibration as excal
import validation as val
import util
import os

# calibration 결과들을 저장할 폴더명
cali_folder = 'use_origin_img5' 
# thermal board 데이터가 있는 폴더 명
thermal_folder = '/media/daheeyoun/DAHEEUSB/rosbag_images/1/thermal'
# rgb board 데이터가 있는 폴더 명
rgb_folder =  '/media/daheeyoun/DAHEEUSB/rosbag_images/1/rgb'
# 캘리브레이션을 위해 임의로 선택할 파일의 갯수(5~60개가 적당)
random_select = 50
os.makedirs(cali_folder, exist_ok=True)

thermal_incal_conf = {
    'cali_folder': cali_folder, 
    'img_folder' : thermal_folder,
    'grid' : (4, 6),
    'interval' : 100.0,
    'img_type' : 'thermal', #[rgb|thermal]
    'circle_color' : 'black', #[black|white]
    'checked' : False,
    'area' : True,
    'minArea' : 100,
    'maxArea' : 500, 
    'circulerity' : 0.8,
    'img_save' : False
}

rgb_incal_conf = {
    'cali_folder': cali_folder, 
    'img_folder' : rgb_folder,
    # 'resize' : (640, 480), # 이미지 resize와 동시에 감지하는 원의 크기 변경
    # 'minArea' : 10,
    # 'maxArea' : 500,
    'grid' : (4, 6),
    'interval' : 100.0,
    'img_type' : 'rgb', #[rgb|thermal]
    'circle_color' : 'white', #[black|white]
    'checked' : False,
    'area' : True,
    'minArea' : 500,
    'maxArea' : 2000, 
    'circulerity' : 0.8,
    'img_save' : False
} 

extrin_conf = {
    'cali_folder': cali_folder, 
    'thermal_file': f'{cali_folder}/calibration_data_with_centers_thermal.npz',
    'rgb_file' : f'{cali_folder}/calibration_data_with_centers_rgb.npz',
    'grid' : (4, 6),
    'interval': 100.0
}

def get_calibration() :
    thermal_cal = incal.IN_CALIBRATION(thermal_incal_conf)
    rgb_cal = incal.IN_CALIBRATION(rgb_incal_conf)

    thermal_cal.find_valid_img()
    rgb_cal.find_valid_img(thermal_cal)

    common_key = util.common_image(thermal_cal, rgb_cal, k=random_select)
    
    thermal_cal.refine_images(common_key)
    rgb_cal.refine_images(common_key)

    thermal_cal.calibration()
    rgb_cal.calibration()

    ex_cal = excal.EX_CALIBRATION(extrin_conf)
    ex_cal()

# rgb 이미지에서 종종 이미지 캡쳐하다가 노이즈가 생기는 경우가 있음
# 이를 수동으로 제거하고 실행.
def get_calibration2() :
    rgb_incal_conf['checked'] = True
    rgb_cal = incal.IN_CALIBRATION(rgb_incal_conf)
    rgb_cal.find_valid_img()
    rgb_cal.calibration()

    ex_cal = excal.EX_CALIBRATION(extrin_conf)

    ex_cal()     

def reprojection_error() :
    thermal_file = f'{cali_folder}/calibration_data_with_centers_thermal.npz'
    rgb_file = f'{cali_folder}/calibration_data_with_centers_rgb.npz'
    extrin_file = f'{cali_folder}/thermal_rgb_extrinsics.npz'

    val.get_all_matrix(extrin_file)

    val.reprojection_error(thermal_file, rgb_file, extrin_file) 

def rectification() :
    thermal_file = f'{cali_folder}/calibration_data_with_centers_thermal.npz'
    rgb_file = f'{cali_folder}/calibration_data_with_centers_rgb.npz'
    extrin_file = f'{cali_folder}/thermal_rgb_extrinsics.npz'

    val.rectified_images(thermal_file, rgb_file, extrin_file, thermal_folder, rgb_folder)

# 내부 파라미터가 정확하게 나왔는지 판단합니다.
def validation_intrinsic_mtx(train_folder, val_folder) :
    print("Thermal Intrinsic Matrix")
    train_file = f'{train_folder}/calibration_data_with_centers_thermal.npz'
    val_file = f'{val_folder}/calibration_data_with_centers_thermal.npz'
    #val.validation_with_other_img(train_file, val_file)
    val.residual_axis_RMSE(train_file, val_file)

    print("RGB Intrinsic Matrix")
    train_file = f'{train_folder}/calibration_data_with_centers_rgb.npz'
    val_file = f'{val_folder}/calibration_data_with_centers_rgb.npz'
    val.residual_axis_RMSE(train_file, val_file)
 
# 랜덤 이미지 셋을 선택하고 여기에 대해 intrinsic/extrinsic matrix 연산
#get_calibration()

# RGB에서 이상 이미지 제거 하고, 이에 대해 intrinsic/extrinsic matrix 재연산
#get_calibration2()

# reprojection error 출력
reprojection_error()

# rectification 연산
# combine.jpg로 결과 이미지 저장
rectification()

file1 = 'use_origin_img'
file2 = 'use_origin_img2'
# 내부 행렬이 과적합 없이 잘 연산 되었는지 검증
validation_intrinsic_mtx(file1, file2)

