# Third-party attribution

AI reference matching (2026-09-22): DY112/Neural-Preset, an **unofficial**
implementation of Neural Preset (CVPR 2023), by Dongyoung Kim.
Source: https://github.com/DY112/Neural-Preset
Pinned revision: 5992c5b48a189051a11813b9b8095229cb032ce2
Code: MIT, see Neural-Preset-LICENSE.txt. Upstream README describes the project
as intended for research purposes only and refers commercial users to the
original work; the code license is not an independent commercial grant for
the pretrained model or its training photography. This integration is for
local evaluation; check model/data rights before commercial redistribution.
Original checkpoint: https://drive.google.com/file/d/1TZRVwIlzBBewwzgjrScrVzeynhBSLmm0/view
The bundled neural-preset.pth contains only net.* tensors, extracted with
weights_only=True (training configuration classes replaced with inert holders).
SHA-256: 7a683ea84fee34ada68b1231b8ac81b3350a926ed2f2a1b6cbd3b9e503da880c
Adaptations: inference-only loader, SHA validation, CPU/CUDA execution,
256x256 RGB encoder input, embedding cache, export of the learned 3x3 transform,
exact identity for identical inputs, adjustable blend/lightness/skin protection.
The model estimates a global linear RGB transform; it does not segment matching
objects or generate spatial content. No training is performed in this application.

EfficientNet-PyTorch 0.7.1 by Luke Melas-Kyriazi, Apache-2.0;
https://github.com/lukemelas/EfficientNet-PyTorch
See EfficientNet-PyTorch-LICENSE.txt. No separate ImageNet weights are downloaded;
the entire encoder is loaded from the Neural-Preset checkpoint above.

Image-Adaptive-3DLUT by Hui Zeng, Jianrui Cai, Lida Li, Zisheng Cao and Lei Zhang.
Source: https://github.com/HuiZeng/Image-Adaptive-3DLUT
Pinned revision: `b491f6df64a588864739a157db271e5c848e1805`.
License: Apache License 2.0; see `Image-Adaptive-3DLUT-LICENSE.txt`.

The classifier architecture in `engine.py` is adapted from `models.py`.
Changes: removed training-only code and custom CUDA extension; used PyTorch
grid_sample for trilinear interpolation; added tiled inference, hash validation,
weights-only deserialization, CPU fallback on CUDA out-of-memory and manual tone controls.
Original trained weights (paired sRGB) are in `models/classifier.pth` and `models/LUTs.pth`.
The demonstration image `samples/demo.jpg` comes from `demo_images/sRGB/a1629.jpg`
in the same repository (MIT-Adobe FiveK example). It is provided for local evaluation;
the repository's code license should not be treated as an independent grant of
rights to redistribute the underlying photography in commercial sample packs.

Paper: Learning Image-adaptive 3D Lookup Tables for High Performance Photo
Enhancement in Real-time. IEEE TPAMI 44(4), 2058–2073, 2022.
https://arxiv.org/abs/2009.14468

Other runtime dependencies: PyTorch (BSD-style), NumPy (BSD-3-Clause), Pillow (MIT-CMU).
Dependency license texts are distributed with their installed packages.

Desktop edition (2026-09-21): PySide6-Essentials and Shiboken 6.11.2
(LGPL-3.0/GPL-3.0/commercial), pillow-heif 1.7.0 (BSD-3-Clause).
pillow-heif's Windows wheel bundles libheif, libde265 and x265 and their respective
license notices. The wheel's LICENSES_bundled.txt is included in the frozen
package's dist-info directory. Qt's GPL and LGPL texts are in this directory;
sources are available at https://github.com/pyside/pyside-setup/tree/v6.11.2
and https://code.qt.io/cgit/qt/qtbase.git/tree/?h=v6.11.2 .
PyInstaller is GPL with its distribution exception; the application is packed
as a directory with dynamically loaded Qt libraries.

Reference-color matching uses independently implemented, bounded Oklab color
statistics. Oklab matrices and color-space definition: Björn Ottosson,
https://bottosson.github.io/posts/oklab/ (MIT license for the published code).

Face and skin-area detection uses OpenCV's bundled Haar cascade when the
desktop dependency is installed. OpenCV is distributed under the Apache 2
license; source and license: https://opencv.org/license/ .
The bundled Haar cascade contains its own Intel license notice in the XML.
The packaged self-test portrait is a NASA photograph of Eileen Collins from
scikit-image; provenance is included in tests/fixtures/README.md.

Face detection (2026-09-22): YuNet, face_detection_yunet_2023mar.onnx,
https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet, MIT License.
SHA-256 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4.

Face parsing (2026-09-22): BiSeNet ResNet18 from https://github.com/yakhyo/face-parsing
(release "weights", resnet18.onnx, stored as models/face_parsing_resnet18.onnx), MIT License.
SHA-256 0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f.
The weights were trained on CelebAMask-HQ, which is released for non-commercial research
purposes only; review that license before any commercial distribution.
