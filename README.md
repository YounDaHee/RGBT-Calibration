# RGB-T-Calibration
성균관대학교 RISE 랩실_RGB-T Calibration 과정

지름 2cm, 10cm 간격의 원형 패턴을 사용하는 캘리브레이션 보드를 탐지하기 위해 작성한 코드입니다.
보드는 나무와 PLA+로 구성되며, PLA+ 재질의 원형 판을 차갑게 만든 뒤, 열을 가한 보드에 조립하는 방식으로 thermal에서의 측정이 이루어집니다.

<img width="320" height="240" alt="rgb_00084" src="https://github.com/user-attachments/assets/a0055b33-a400-4f7a-ace4-ef43ac28bd6d" />
<img width="320" height="240" alt="thermal_00058" src="https://github.com/user-attachments/assets/cd6f7ad2-53e8-45ad-bc1b-e6e7e50d4610" />

run.py 를 실행해서 캘리브레이션 코드를 실행할 수 있습니다.
해당 코드의 다양한 기능들은 run.py의 함수를 통해 사용할 수 있습니다.

- 코드를 실행하기 전 run.py에서 이미지 폴더 설정을 확인하는 것을 잊지 마십쇼
- 이 코드는 두 카메라의 같은 시점의 이미지가 같은 파일명(숫자)를 가지고 있다는 걸 전재로 합니다.
- 두 카메라의 이미지는 각각 'rgb_숫자.png', 'thermal_숫자.png' 형태도 저장되어야 정상적으로 인식 됩니다.

두 이미지의 사이즈가 다를 경우

1) intrinsic 결과를 사이즈에 맞게 스케일링하여 사용
2) calibration 이전 이미지 사이즈를 변경하여 사용
   
두가지 방법이 존재한다. 

이는 run.py에 있는 configuration을 조절해서 제어할 수 있다.

<img width="330" height="180" alt="image" src="https://github.com/user-attachments/assets/43b112cc-7507-47f7-823c-060868996c44" />

가끔 이상하게 캡쳐된 이미지가 캘리브레이션에 사용되어 결과에 악영향을 미칠 수 있는데,

이는 캘리브레이션 결과 폴더에 생성되는 selected_img 폴더에서 해당 이미지를 제거해서 run.py의 calibration2 함수를 사용하는 것으로 극복할 수 있다.

정상적으로 측정된 캘리브레이션 데이터에 대해서 다음과 같은 combine.png 파일이 생성됩니다.

<img width="320" height="480" alt="image" src="https://github.com/user-attachments/assets/8a10a905-91ef-40b8-b5f5-c3085eb06bd3" />
