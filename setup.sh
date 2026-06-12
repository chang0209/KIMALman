#!/bin/bash
# 필수 라이브러리 설치 (v2.3)
pip install -q opencv-python Pillow numpy deep-translator manga-ocr gradio

wget https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf -O /content/NanumGothicBold.ttf
