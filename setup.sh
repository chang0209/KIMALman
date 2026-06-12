#!/bin/bash
# 필수 라이브러리 설치 (manga-ocr 라이브러리가 진짜 핵심입니다!)
pip install -q opencv-python Pillow numpy deep-translator manga-ocr gradio

# 수정된 폰트 다운로드 URL
wget https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf -O /content/NanumGothicBold.ttf
