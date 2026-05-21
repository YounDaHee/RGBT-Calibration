# RGB-T-Calibration
성균관대학교 RISE 랩실_RGB-T Calibration 과정

지름 2cm, 10cm 간격의 원형 패턴을 사용하는 캘리브레이션 보드를 탐지하기 위해 작성한 코드입니다.
캘리브레이션 보드는 나무와 PLA+로 구성되어 있습니다.
1. PLA+ 재질의 원형 판을 차갑게 만든 뒤 조립
2. 햇빛이나 난로를 통해 보드의 열을 고루 뎁힌 뒤 분리한 원형 판을 조립
두가지 방식을 이용하여 열 분포를 다르게 하여 캘리브레이션을 수행할 수 있습니다.

<img width="320" height="240" alt="rgb_00084" src="https://github.com/user-attachments/assets/a0055b33-a400-4f7a-ace4-ef43ac28bd6d" />
<img width="320" height="240" alt="thermal_00058" src="https://github.com/user-attachments/assets/cd6f7ad2-53e8-45ad-bc1b-e6e7e50d4610" />

run.py 를 실행해서 캘리브레이션 코드를 실행할 수 있습니다.
해당 코드의 다양한 기능들은 run.py의 함수를 통해 사용할 수 있습니다.

- 이 코드는 두 카메라의 같은 시점의 이미지가 같은 파일명(숫자)를 가지고 있다는 걸 전재로 합니다.
- 두 카메라의 이미지는 '숫자.png' 형태도 저장되어야 정상적으로 인식 됩니다.

config.json 폴더의 내용을 수정하여 횐경에 맞는 파라미터를 설정할 수 있습니다.
```
{
   # 결과가 저장될 폴더 위치
    "cali_folder": "example_calibration",
   # 온도 카메라에서 인식한 보드의 이미지가 저장된 폴더 위치
    "thermal_folder": "example_data/example_thermal",
   # 
    "rgb_folder": "example_data/example_rgb",
    "grid": [4, 6],
    "interval": 0.100, 
    "thermal": {
        "area": true,
        "minArea": 200,
        "maxArea": 4000,
        "circulerity": 0.8,
        "img_save": false
    },
    "rgb": {
        "area": true,
        "minArea": 1000,
        "maxArea": 120000,
        "circulerity": 0.8,
        "img_save": false
    },
    "depth" : 2,
    "rgb_path" : "example_data/example_rgb/2000000.png",
    "th_path" : "example_data/example_thermal/2000000.png"
}
```

테스트 수행을 위해 실험실 환경에서 촬영한 보드 데이터를 정리하였습니다.

