# 개발환경

- OS : 우분투 18.04.5 LTS
- conda 가상환경 만들어서 conda 명령어로 순서대로 설치
  - python == 3.6.5
  - tensorflow-gpu == 1.15
  - scikit-learn 최신
  - scikit-image 최신
  - cudatoolkit == 10.0.130(tensorflow-gpu 1.15설치시 자동 설치)
  - cudnn == 7.6.5(tensorflow-gpu 1.15설치시 자동 설치)
  - Pillow
  - cython
  - keras == 2.3.0(pip로 설치)
  - tqdm
- 모두 설치 후, integrated_main폴더에서 터미널로 python main.py 실행
