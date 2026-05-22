# RGB-T-Calibration
RGB-T Calibration process from the RISE Laboratory at Sungkyunkwan University

This project provides a calibration pipeline between RGB and Thermal cameras using a circular calibration board pattern.

The calibration board uses:

2 cm diameter circles
10 cm spacing between circles

The board is designed to generate thermal contrast for stable detection in thermal images.

# Calibration Board Construction

The calibration board consists of:

- Wooden base plate
- PLA+ circular plates

Two methods can be used to generate thermal contrast.

1. Cooling Method : 
   Cool down the PLA+ circular plates
   Assemble them onto the board
   Detect temperature differences from the background
2. Heating Method : 
   Heat the board uniformly using sunlight or a heater
   Reassemble the separated PLA+ circles
   Use thermal contrast for circle detection

These approaches improve circle detection performance in thermal images.

<img width="911" height="1362" alt="image" src="https://github.com/user-attachments/assets/30cb3523-1165-4bcd-b77e-8d5f2fb41c05" />

# Project Execution

Run the calibration pipeline using:
```
python3 run.py --config config.json --mode all
```

# Important Notes
- RGB and Thermal images must be captured at the same timestamp.
- Image filenames from both cameras must match.
  
```
{
    // Directory where calibration results will be saved
    "cali_folder": "example_calibration",

    // Directory containing detected board images from the thermal camera
    "thermal_folder": "example_data/example_thermal",

    // Directory containing detected board images from the RGB camera
    "rgb_folder": "example_data/example_rgb",

    // Calibration board pattern size [rows, columns]
    "grid": [4, 6],

    // Distance between adjacent circles on the board (meter unit)
    "interval": 0.100,

    "thermal": {
        // Enable area-based blob filtering
        "area": true,

        // Minimum blob area for thermal image detection
        "minArea": 200,

        // Maximum blob area for thermal image detection
        "maxArea": 4000,

        // Minimum circularity threshold for blob detection
        "circulerity": 0.8,

        // Save intermediate detection result images
        "img_save": false
    },

    "rgb": {
        // Enable area-based blob filtering
        "area": true,

        // Minimum blob area for RGB image detection
        "minArea": 1000,

        // Maximum blob area for RGB image detection
        "maxArea": 120000,

        // Minimum circularity threshold for blob detection
        "circulerity": 0.8,

        // Save intermediate detection result images
        "img_save": false
    },

    // Reference depth value used for final calibration validation
    "depth": 2,

    // RGB image path used for validation
    "rgb_path": "example_data/example_rgb/2000000.png",

    // Thermal image path used for validation
    "th_path": "example_data/example_thermal/2000000.png"
}
```

# Output files
Calibration results are stored as .npz files.

1. Intrinsic Calibration Result
 
  ```
 `camera_matrix`      Camera intrinsic matrix      
 `distortion_coeffs`  Lens distortion coefficients 
 `rvecs`              Rotation vectors             
 `tvecs`              Translation vectors          
 `corner_storage`     Detected circle centers      
 `img_size`           Image resolution            
 `reproj_error`       Mean reprojection error      

  ```
  
2. Extrinsic Calibration Result
   
  ```
 `R`             Rotation matrix between RGB and Thermal cameras    
 `T`             Translation vector between RGB and Thermal cameras 
 `E`             Essential matrix                                   
 `F`             Fundamental matrix                                 
 `reproj_error`  Stereo calibration reprojection error              
 `matched_ids`   Matched image IDs                                  

  ```
