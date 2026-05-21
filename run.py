import intrinsic_calibration as incal
import extrinsic_calibration as excal
import calibration_validation as val
import util

import os
import numpy as np
import argparse
import json
import cv2 as cv
from pathlib import Path

def setting_configuration(config_path, mode) : 
    config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
    else:
        print("[Error] 파일이 존재하지 않습니다.")
        exit(1)

    cali_folder = loaded_config["cali_folder"]
    rgb_folder = loaded_config["rgb_folder"]
    grid = loaded_config["grid"]
    interval = loaded_config["interval"]

    configs = {}
    os.makedirs(cali_folder, exist_ok=True)
    if mode == 'all' or mode == "thermal" :
        thermal_folder = loaded_config["thermal_folder"]
        thermal_incal_conf = {
            'cali_folder': cali_folder, 
            'img_folder' : thermal_folder,
            'grid' : grid,
            'interval' : interval,
            'img_type' : 'thermal', #[rgb|thermal]
            'circle_color' : 'black', #[black|white]
            'area' : loaded_config["thermal"]["area"],
            'minArea' : loaded_config["thermal"]["minArea"],
            'maxArea' : loaded_config["thermal"]["maxArea"], 
            'circulerity' : loaded_config["thermal"]["circulerity"],
            'img_save' : loaded_config["thermal"]["img_save"],
            'file': f'{cali_folder}/calibration_data_with_centers_thermal.npz'
        }
        configs['th_incal'] = thermal_incal_conf

    if mode == 'all' or mode == "rgb" :
        rgb_incal_conf = {
            'cali_folder': cali_folder, 
            'img_folder' : rgb_folder,
            'grid' : grid,
            'interval' : interval,
            'img_type' : 'rgb', #[rgb|thermal]
            'circle_color' : 'white', #[black|white]
            'area' : loaded_config["rgb"]["area"],
            'minArea' : loaded_config["rgb"]["minArea"],
            'maxArea' : loaded_config["rgb"]["maxArea"], 
            'circulerity' : loaded_config["rgb"]["circulerity"],
            'img_save' : loaded_config["rgb"]["img_save"],
            'file': f'{cali_folder}/calibration_data_with_centers_rgb.npz'
        }
        configs['rgb_incal'] = rgb_incal_conf

    if mode == "all" :
        extrin_conf = {
            'cali_folder': cali_folder, 
            'thermal_file': f'{cali_folder}/calibration_data_with_centers_thermal.npz',
            'rgb_file' : f'{cali_folder}/calibration_data_with_centers_rgb.npz',
            'file': f'{cali_folder}/thermal_rgb_extrinsics.npz',
            'grid' : grid,
            'interval': interval
        }
        configs['excal'] = extrin_conf

        validation = {
            'cali_folder' : cali_folder,
            'depth' : loaded_config["depth"],
            'rgb_path' : loaded_config["rgb_path"],
            'th_path' : loaded_config["th_path"]
        }

    return configs, validation

def rgb_calibration(rgb_incal_conf) :
    rgb_cal = incal.IN_CALIBRATION(rgb_incal_conf, unit=True)

    rgb_cal.find_valid_img()

    common_key = util.common_image(rgb_cal, k=-1)
    
    rgb_cal.refine_images(common_key)

    rgb_cal.calibration()

    rgb_file = rgb_incal_conf["file"]
    rgb_data = np.load(rgb_file, allow_pickle=True)
    
    print("\n[RGB Camera-Matrix]")
    print(rgb_data['camera_matrix'])
    print("[RGB Distortion-Vector]")
    print(rgb_data['distortion_coeffs'])
    print(f"RGB Reprojection Error : {rgb_data['reproj_error']}" )

def thermal_calibration(thermal_incal_conf) :
    thermal_cal = incal.IN_CALIBRATION(thermal_incal_conf, unit=True)

    thermal_cal.find_valid_img()

    common_key = util.common_image(thermal_cal, k=-1)
    
    thermal_cal.refine_images(common_key)

    thermal_cal.calibration()

    thermal_file = thermal_incal_conf["file"]
    thermal_data = np.load(thermal_file, allow_pickle=True)
    
    print("\n[Thermal Camera-Matrix]")
    print(thermal_data['camera_matrix'])
    print("[Thermal Distortion-Vector]")
    print(thermal_data['distortion_coeffs'])
    print(f"Thermal Reprojection Error : {thermal_data['reproj_error']}" )

def extrainsic_caligration(configs) :
    thermal_cal = incal.IN_CALIBRATION(configs["th_incal"])
    rgb_cal = incal.IN_CALIBRATION(configs["rgb_incal"])

    thermal_cal.find_valid_img()
    rgb_cal.find_valid_img(thermal_cal)

    common_key = util.common_image(thermal_cal, rgb_cal, k=-1)
    
    thermal_cal.refine_images(common_key)
    rgb_cal.refine_images(common_key)

    ex_cal = excal.EX_CALIBRATION(configs["excal"])
    ex_cal(thermal_calibration=thermal_cal, rgb_calibration=rgb_cal)

    ex_file = configs["excal"]["file"]
    ex_data = np.load(ex_file, allow_pickle=True)

    print("\n[Rotation-Matrix]")
    print(ex_data['R'])
    print("\n[Translation-Vector]")
    print(ex_data['T'])
    print(f"Thermal Reprojection Error : {ex_data['reproj_error']}" )

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGB-T calibration runner")
    parser.add_argument(
        "--config",
        help="Path to configuration json file.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "thermal", "rgb"],
        default="all",
        help="Calibration mode to run.",
    )
    args = parser.parse_args()

    configs, validation = setting_configuration(args.config, args.mode)
    if args.mode == "thermal" or args.mode == "all" : 
        thermal_calibration(configs["th_incal"])
        undist_th = val.img_undistort(configs["th_incal"], validation["th_path"])
        cv.imwrite(f"{configs['th_incal']['cali_folder']}/undistort_thermal.jpg",undist_th)
    if args.mode == "rgb" or args.mode == "all" : 
        rgb_calibration(configs["rgb_incal"])
        undist_rgb = val.img_undistort(configs["rgb_incal"], validation["rgb_path"])
        cv.imwrite(f"{configs['rgb_incal']['cali_folder']}/undistort_rgb.jpg",undist_rgb)
    if args.mode == "all" :
        extrainsic_caligration(configs)
        val.img_overlay(configs, validation)
