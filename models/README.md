# 모델 준비

모델 가중치는 GitHub 저장소에 포함하지 않습니다. 직접 내려받은 파일을
이 폴더에 저장하거나, 기존 온빛 설치본의 `_internal/models/`에서 복사하세요.
프로그램은 아래 SHA-256을 확인하므로 파일 이름만 같은 다른 가중치는 사용할 수 없습니다.

| 저장할 파일 | 원본 |
| --- | --- |
| `classifier.pth` | [Image-Adaptive-3DLUT sRGB paired](https://github.com/HuiZeng/Image-Adaptive-3DLUT/blob/b491f6df64a588864739a157db271e5c848e1805/pretrained_models/sRGB/classifier.pth) |
| `LUTs.pth` | [Image-Adaptive-3DLUT sRGB paired](https://github.com/HuiZeng/Image-Adaptive-3DLUT/blob/b491f6df64a588864739a157db271e5c848e1805/pretrained_models/sRGB/LUTs.pth) |
| `face_detection_yunet_2023mar.onnx` | [OpenCV YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `face_parsing_resnet18.onnx` | [face-parsing weights](https://github.com/yakhyo/face-parsing/releases/tag/weights)의 `resnet18.onnx`를 이름 변경 |
| `neural-preset.pth` | 아래 Neural-Preset 변환 절차 |

GitHub에서 모델을 내려받을 때 Git LFS 포인터 텍스트 대신 실제 바이너리를 다운로드하세요.
각 프로젝트의 모델·학습 데이터 이용 조건은 [출처와 라이선스](../third_party/NOTICE.md)를 확인하세요.
특히 얼굴 파싱의 CelebAMask-HQ 데이터와 비공식 Neural-Preset 프로젝트는 연구 목적 관련 조건이 있습니다.

## Neural-Preset

1. 프로젝트 루트에서 `setup.ps1`을 실행해 PyTorch 2.7.1 개발 환경을 준비합니다.
2. [DY112/Neural-Preset](https://github.com/DY112/Neural-Preset)의
   [공개 best.ckpt](https://drive.google.com/file/d/1TZRVwIlzBBewwzgjrScrVzeynhBSLmm0/view)를 직접 내려받습니다.
3. 아래 명령으로 설정·학습 메타데이터를 제거하고 텐서만 추출합니다.

```powershell
.\.venv\Scripts\python.exe tools/convert_neural_preset.py "C:\path\to\best.ckpt"
```

원본 SHA-256은 `39f5c14af0aa4140695781c8c52f13add3836da271b61a8fd6ade9fd4fddc5ff`입니다.
변환기는 원본 해시를 먼저 검증하고 `weights_only=True`로 읽습니다.
기존 출력 파일이 있으면 덮어쓰지 않습니다. 앱 실행 중 모델 다운로드나 사진 업로드는 없습니다.

## 설치 후 SHA-256

```text
classifier.pth  bae9865395625ecae58cfe86147e521093bb1e29e7b2544e02adb238b8035021
LUTs.pth  c1bb2bc4b7239c1a7e96159f5923123ba796b1fceb0b8c3132b423ea825b821a
face_detection_yunet_2023mar.onnx  8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
face_parsing_resnet18.onnx  0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f
neural-preset.pth  7a683ea84fee34ada68b1231b8ac81b3350a926ed2f2a1b6cbd3b9e503da880c
```

PowerShell의 `Get-FileHash models/* -Algorithm SHA256`으로 확인할 수 있습니다.
모델을 준비한 뒤 `python desktop.py`로 실행하거나 `build.ps1`로 EXE를 만듭니다.
전체 테스트와 빌드에는 [샘플 사진](../samples/README.md)도 필요합니다.
