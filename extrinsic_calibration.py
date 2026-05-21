import cv2
import numpy as np
import os

class EX_CALIBRATION :
    
    def __init__(self, conf) :
        self.pattern_size = conf['grid']
        self.interval = conf['interval']
        self.save_data_path = conf['file']
        self.thermal_file = conf['thermal_file']
        self.rgb_file = conf['rgb_file']
        self.result_folder = conf['cali_folder']

    def _load_intrinsic_data(self, npz_file):
        if npz_file is None:
            raise ValueError("npz file must be provided for extrinsic calibration.")

        data = np.load(npz_file, allow_pickle=True)
        return {
            'img_size': tuple(data['img_size']),
            'camera_matrix': data['camera_matrix'],
            'distortion_coeffs': data['distortion_coeffs']
        }

    def _load_point_data(self, calibration_obj):
        return {
            'img_size': calibration_obj.img_shape,
            'corner_storage': calibration_obj.center_storage,
        }

    def __call__(self, 
                 thermal_calibration, rgb_calibration,
                 mtx_thermal=None, dist_thermal=None,
                 mtx_rgb=None, dist_rgb=None) :
        thermal_intrinsic = self._load_intrinsic_data(self.thermal_file)
        rgb_intrinsic = self._load_intrinsic_data(self.rgb_file)
        thermal_data = self._load_point_data(thermal_calibration)
        rgb_data = self._load_point_data(rgb_calibration)

        # 해상도가 다른 두 이미지를 스케일링 하기 위해 사용
        th_w, th_h = thermal_data['img_size']
        rgb_w, rgb_h = rgb_data['img_size']

        scale_x = 1.0
        scale_y = 1.0
        if(th_w!=rgb_w) :
            scale_x = th_w / rgb_w
            scale_y = th_h / rgb_h

        if(mtx_thermal is None) : mtx_thermal = thermal_intrinsic['camera_matrix']
        if(dist_thermal is None) : dist_thermal = thermal_intrinsic['distortion_coeffs']
        if(mtx_rgb is None) : mtx_rgb = rgb_intrinsic['camera_matrix']
        if(dist_rgb is None) : dist_rgb = rgb_intrinsic['distortion_coeffs']

        mtx_thermal = np.array(mtx_thermal, copy=True)
        dist_thermal = np.array(dist_thermal, copy=True)
        mtx_rgb = np.array(mtx_rgb, copy=True)
        dist_rgb = np.array(dist_rgb, copy=True)
        
        if(th_w!=rgb_w) :
            mtx_rgb[0, 0] *= scale_x  # fx
            mtx_rgb[0, 2] *= scale_x  # cx
            mtx_rgb[1, 1] *= scale_y  # fy
            mtx_rgb[1, 2] *= scale_y  # cy

        thermal_img = {fname for fname in thermal_data['corner_storage'].keys()}
        rgb_img = {fname for fname in rgb_data['corner_storage'].keys()}

        common_keys = thermal_img & rgb_img

        thermal_imgpoints_dict = {k:thermal_data['corner_storage'][k]
                                    for k in [cname for cname in common_keys]}
        rgb_imgpoints_dict = {k:rgb_data['corner_storage'][k]
                                    for k in [cname for cname in common_keys]}
        
        def get_frame_id(fname):
            name, _ = os.path.splitext(fname)
            return name.split('_')[-1]

        thermal_imgpoints_dict = {os.path.basename(k): v for k, v in thermal_imgpoints_dict.items()}
        rgb_imgpoints_dict = {os.path.basename(k): v for k, v in rgb_imgpoints_dict.items()}

        thermal_by_id = {get_frame_id(k): v for k, v in thermal_imgpoints_dict.items()}
        rgb_by_id = {get_frame_id(k): v for k, v in rgb_imgpoints_dict.items()}
       
        objp = np.zeros((self.pattern_size[0]*self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.interval

        objpoints = []
        imgpoints_thermal = []
        imgpoints_rgb = []
        matched_ids = []

        for fid in sorted(thermal_by_id.keys()):
            if fid in rgb_by_id:
                objpoints.append(objp)
                imgpoints_thermal.append(thermal_by_id[fid])
                if(th_w!=rgb_w) :
                    rgb_by_id[fid][..., 0] *= scale_x
                    rgb_by_id[fid][..., 1] *= scale_y
                imgpoints_rgb.append(rgb_by_id[fid])
                matched_ids.append(fid)

        print(f"[INFO] Matched {len(objpoints)} pairs.")
        if len(objpoints) == 0:
            raise RuntimeError("No matching pairs found.")

        flags = cv2.CALIB_FIX_INTRINSIC 
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-8)

        ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_rgb, imgpoints_thermal, 
            mtx_rgb, dist_rgb,
            mtx_thermal, dist_thermal,
            thermal_data['img_size'],
            criteria=criteria,
            flags=flags
        )

        np.savez(
            self.save_data_path,
            R=R, T=T, E=E, F=F,
            reproj_error=ret,
            matched_ids=matched_ids,
            mtx1 = mtx_rgb,
            dist1 = dist_rgb,
            img_size1 = rgb_data['img_size'],
            mtx2 = mtx_thermal,
            dist2 = dist_thermal,
            img_size2 = thermal_data['img_size']
        )

        print(f"\n[INFO] Calibration data saved to:\n{self.save_data_path}")
