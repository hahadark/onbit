# 개발용 샘플 사진

샘플 사진과 그 보정 결과는 Git에 포함하지 않습니다.
뷰어의 샘플 버튼, 기존 테스트, EXE 빌드에는 `samples/demo.jpg`가 필요합니다.

테스트와 같은 사진을 사용하려면 Image-Adaptive-3DLUT 저장소의
[a1629.jpg](https://github.com/HuiZeng/Image-Adaptive-3DLUT/blob/b491f6df64a588864739a157db271e5c848e1805/demo_images/sRGB/a1629.jpg)를
직접 내려받아 이 폴더의 `demo.jpg`로 저장하세요. MIT-Adobe FiveK 사진이며
코드의 Apache 라이선스가 사진 자체의 배포 권한을 뜻하지는 않습니다.

로컬 뷰어용으로는 직접 소유한 사진을 `demo.jpg`로 저장해도 됩니다.
다만 기존 테스트의 수치 기준은 위 원본 샘플을 기준으로 합니다.
NASA 인물 테스트 사진은 출처 안내와 함께 `tests/fixtures/`에 포함됩니다.
