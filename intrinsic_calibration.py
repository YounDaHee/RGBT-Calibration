import cv2
import numpy as np
import glob
import os
import util

class IN_CALIBRATION :
    
    def __init__(self, conf, unit=False):
        self.img_folder = conf['img_folder']
        self.pattern_size = conf['grid']
        self.interval = conf['interval']
        self.type = conf['img_type']
        self.result_dir = conf['cali_folder']
        self.save_data_path = conf['file']
        self.img_save = conf['img_save']

        self.unit = unit

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = conf['area'] #True
        if conf['area'] :
            params.minArea = conf['minArea'] #1000     
            params.maxArea = conf['maxArea'] #50000  

        params.filterByCircularity = True
        params.minCircularity = conf['circulerity']

        params.filterByColor = True
        if "white" in conf['circle_color']:
            # 원의 내부가 흰색이어야 함
            params.blobColor = 255
        else :
            # 원의 내부가 검은색이어야 함
            params.blobColor = 0
            params.minThreshold = 10
            params.maxThreshold = 100
            params.thresholdStep = 5

        params.filterByConvexity = False
        params.filterByInertia = False

        ver = cv2.__version__.split('.')
        if int(ver[0]) < 3:
            self.detector = cv2.SimpleBlobDetector(params)
        else:
            self.detector = cv2.SimpleBlobDetector_create(params)

        if 'resize' in conf.keys() :
            self.re_size = conf['resize']
            
    def refine_images(self, common_keys) :
        self.center_storage = {k:self.center_storage[k] 
                                    for k in [cname for cname in common_keys]}
        
        if self.img_save : 
            output_dir = f'{self.result_dir}/selected_img_{self.type}'
            os.makedirs(output_dir, exist_ok=True)
        for k in self.center_storage.keys() :
            img = cv2.imread(f'{self.img_folder}/{k}', cv2.IMREAD_GRAYSCALE)

            img_blob = cv2.drawKeypoints(
                img, self.detector.detect(img), None, (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
        
            cali_img = cv2.drawChessboardCorners(img_blob, self.pattern_size, self.center_storage[k], True)
            cali_img = util.draw_points_with_index(cali_img, self.center_storage[k])
            if self.img_save : 
                save_name = os.path.join(output_dir, f"centers_{k}")
                cv2.imwrite(save_name, cali_img)
        

    def find_valid_img(self, THERMAL = None) :
        # Prepare object points
        # 가상의 3차원 월드 좌표계 생성

        # Arrays for calibration
        self.center_storage = {}
        self.img_shape = None

        if 'rgb' == self.type and self.unit == False:
            if THERMAL != None:
                # thermal 상에서 인식한 이미지에 대해서만 연산
                proved_image = THERMAL.center_storage.keys()
            
            image_paths = []
            for detected_image in proved_image :
                image_paths += glob.glob(f'{self.img_folder}/{detected_image}') 
        else :   
            image_paths = glob.glob(f'{self.img_folder}/*.png')

        if self.img_save :
            output_dir = f'{self.result_dir}/valid_img_{self.type}'
            os.makedirs(output_dir, exist_ok=True)

        print(f"[INFO] Found {len(image_paths)} images.")

        succes_cnt = 0
        for fname in image_paths:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if 're_size' in vars(self):
                img = cv2.resize(img, self.re_size)
                
            if self.type == 'thermal' :
                img = util.preprocessing(img)

            ret, center = cv2.findCirclesGrid(
                img, self.pattern_size,
                flags=cv2.CALIB_CB_SYMMETRIC_GRID|cv2.CALIB_CB_CLUSTERING,
                blobDetector=self.detector
            )

            if ret:
                # if not util.legal_calibration(self.pattern_size, center) :
                #     print(f"[WARN] The calibration results are not valid in {os.path.basename(fname)}")
                #     continue
                
                image_name = os.path.basename(fname)
               
                center = util.remove_flip(self.pattern_size, center)
                
                self.center_storage[image_name] = center

                
                if self.img_save :
                    # Draw centers and save
                    img_blob = cv2.drawKeypoints(
                        img, self.detector.detect(img), None, (0, 0, 255),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                    )
                
                    cali_img = cv2.drawChessboardCorners(img_blob, self.pattern_size, center, ret)
                    save_name = os.path.join(output_dir, f"centers_{image_name}")
                    cv2.imwrite(save_name, cali_img)
                succes_cnt += 1
        
        img_h, img_w = img.shape
        self.img_shape = (img_w, img_h)
        print(f'{succes_cnt}/{len(image_paths)}')

    def calibration(self) :
        objp = np.zeros((self.pattern_size[0]*self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.interval

        objpoints = [objp for i in range(len(self.center_storage))]
        
        imgpoints = list(self.center_storage.values())
            
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        # Calibrate
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, self.img_shape, None, None, criteria=criteria 
        ) 

        np.savez(
            self.save_data_path,
            camera_matrix=self.mtx,
            distortion_coeffs=self.dist,
            rvecs=rvecs,
            tvecs=tvecs,
            corner_storage=self.center_storage,
            img_size = self.img_shape,
            reproj_error = util.reprojection_error(objpoints, imgpoints, rvecs, tvecs, self.mtx, self.dist)
        )

        print(f"\n[INFO] Calibration data saved to:\n{self.save_data_path}")
