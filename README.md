# RGB-T-Calibration
성균관대학교 RISE 랩실_RGB-T Calibration 과정

지름 2cm, 10cm 간격의 원형 패턴을 사용하는 캘리브레이션 보드를 탐지하기 위해 작성한 코드입니다.
보드는 나무와 PLA+로 구성되며, PLA+ 재질의 원형 판을 차갑게 만든 뒤, 열을 가한 보드에 조립하는 방식으로 thermal에서의 측정이 이루어집니다.
<img width="1920" height="1080" alt="rgb_00084" src="https://github.com/user-attachments/assets/a0055b33-a400-4f7a-ace4-ef43ac28bd6d" />

<img width="640" height="480" alt="thermal_00058" src="https://github.com/user-attachments/assets/cd6f7ad2-53e8-45ad-bc1b-e6e7e50d4610" />

run.py 를 실행해서 캘리브레이션 코드를 실행할 수 있습니다.
해당 코드의 다양한 기능들은 run.py의 함수를 통해 사용할 수 있습니다.

- 코드를 실행하기 전 run.py에서 이미지 폴더 설정을 확인하는 것을 잊지 마십쇼
- 이 코드는 두 카메라의 같은 시점의 이미지가 같은 파일명(숫자)를 가지고 있다는 걸 전재로 합니다.
- 두 카메라의 이미지는 각각 'rgb_숫자.png', 'thermal_숫자.png' 형태도 저장되어야 정상적으로 인식 됩니다.
