import cv2
import numpy as np
import os

class EX_CALIBRATION :
    
    def __init__(self, conf) :
        self.pattern_size = conf['grid']
        self.interval = conf['interval']

        self.thermal_file = conf['thermal_file']
        self.rgb_file = conf['rgb_file']
        self.result_folder = conf['cali_folder']
        
    def __call__(self) :
        thermal_data = np.load(self.thermal_file, allow_pickle=True)
        rgb_data = np.load(self.rgb_file, allow_pickle=True)

        # 해상도가 다른 두 이미지를 스케일링 하기 위해 사용
        th_w, th_h = thermal_data['img_size']
        rgb_w, rgb_h = rgb_data['img_size']

        if(th_w!=rgb_w) :
            scale_x = th_w / rgb_w
            scale_y = th_h / rgb_h
        
        mtx_thermal = thermal_data['camera_matrix']
        dist_thermal = thermal_data['distortion_coeffs']

        mtx_rgb = rgb_data['camera_matrix']
        if(th_w!=rgb_w) :
            mtx_rgb[0, 0] *= scale_x  # fx
            mtx_rgb[0, 2] *= scale_x  # cx
            mtx_rgb[1, 1] *= scale_y  # fy
            mtx_rgb[1, 2] *= scale_y  # cy
        dist_rgb = rgb_data['distortion_coeffs']

        thermal_img = {fname.split("_")[-1] for fname in thermal_data['corner_storage'].item().keys()}
        rgb_img = {fname.split("_")[-1] for fname in rgb_data['corner_storage'].item().keys()}

        common_keys = thermal_img & rgb_img

        thermal_imgpoints_dict = {k:thermal_data['corner_storage'].item()[k] 
                                    for k in ['thermal_' + cname for cname in common_keys]}
        rgb_imgpoints_dict = {k:rgb_data['corner_storage'].item()[k] 
                                    for k in ['rgb_' + cname for cname in common_keys]}
        
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
            imgpoints_thermal, imgpoints_rgb,
            mtx_thermal, dist_thermal,
            mtx_rgb, dist_rgb,
            thermal_data['img_size'],
            #criteria=criteria,
            flags=flags
        )

        np.savez(
            f"{self.result_folder}/thermal_rgb_extrinsics.npz",
            R=R, T=T, E=E, F=F,
            reproj_error=ret,
            matched_ids=matched_ids,
            mtx1 = mtx1,
            dist1 = dist1,
            mtx2 = mtx2,
            dist2 = dist2
        )


