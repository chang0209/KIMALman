import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deep_translator import GoogleTranslator
from manga_ocr import MangaOcr
import gradio as ui
import google.generativeai as genai
from google.colab import userdata

class RealMangaTranslatorV27:
    def __init__(self, font_path="/content/NanumGothicBold.ttf"):
        self.font_path = font_path
        print("🤖 AI 비전 엔진(Manga-OCR) 로딩 중...")
        self.mocr = MangaOcr()
        self.google_translator = GoogleTranslator(source='ja', target='ko')
        
        # 🔑 Gemini API 설정
        try:
            self.gemini_api_key = userdata.get('GEMINI_API_KEY')
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=(
                    "너는 일본 소년 만화(블리치 등) 전문 번역가야. "
                    "1. '경화수월', '참백도' 같은 만화 고유명사는 철저히 유지해."
                    "2. OCR 인식 오류로 이상한 글자가 섞여 있어도 문맥을 추론해서 완벽한 한국어 대사로 고쳐."
                    "3. 다른 설명 없이 번역된 대사 결과물만 출력해."
                ),
                generation_config={"temperature": 0.2}
            )
            print("✅ Gemini API 연결 성공!")
        except Exception as e:
            print("⚠️ GEMINI_API_KEY를 찾을 수 없거나 이름이 다릅니다. 기본 구글 번역기로 작동합니다.")
            self.model = None
            
        print("✅ 시스템 로딩 완료!")

    def translate_with_llm(self, japanese_text):
        if self.model is None:
            return self.google_translator.translate(japanese_text)
        try:
            response = self.model.generate_content(japanese_text)
            return response.text.strip()
        except Exception as e:
            return self.google_translator.translate(japanese_text)

    def check_constraints(self, img):
        h, w = img.shape[:2]
        if w < 500 or h < 500:
            return False, f"❌ [번역 불가] 이미지 화질이 너무 낮습니다. (최소: 500x500)"
        if w / h > 1.1:
            return False, f"❌ [번역 불가] 한 페이지 형식의 이미지만 지원합니다."
        return True, "정상"

    def detect_speech_bubbles(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bubbles = []
        img_h, img_w = img.shape[:2]
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # [수정 1] 최소 크기 제한 (너무 작은 노이즈 제거)
            if w < 35 or h < 35:
                continue
                
            # [수정 2] 크기 상한선 대폭 완화 (화면의 85% 크기까지 대형 말풍선 허용)
            if w > img_w * 0.75 or h > img_h * 0.85:
                continue
                
            # [수정 3] ✨핵심 필터: 일본 만화 특성 반영 (세로형 말풍선만 통과)
            # 가로가 세로보다 지나치게 넓은 박스(옷, 배경, 얼굴 명암)는 원천 차단
            if w / h > 1.1:
                continue
                
            roi = thresh[y:y+h, x:x+w]
            white_ratio = np.sum(roi == 255) / (w * h)
            
            # [수정 4] 흰색 밀도 기준 최적화 (글씨가 채워진 말풍선은 보통 65%~95% 사이)
            if 0.65 <= white_ratio <= 0.96:
                bubbles.append({"box": (x, y, w, h)})
                
        return bubbles

    def clean_japanese_text(self, cropped_bubble):
        gray = cv2.cvtColor(cropped_bubble, cv2.COLOR_BGR2GRAY)
        _, text_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        text_mask = cv2.dilate(text_mask, kernel, iterations=1)
        cleaned = cv2.inpaint(cropped_bubble, text_mask, 3, cv2.INPAINT_TELEA)
        return cleaned

    def wrap_text(self, text, font, max_width):
        lines = []
        current_line = ""
        for char in text:
            test_line = current_line + char
            bbox = font.getbbox(test_line)
            line_width = bbox[2] - bbox[0]
            if line_width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = char
        if current_line:
            lines.append(current_line)
        return lines

    def optimize_layout(self, text, bubble_w, bubble_h):
        max_text_width = bubble_w - int(bubble_w * 0.25)
        max_text_height = bubble_h - int(bubble_h * 0.25)
        
        best_font_size = 14
        best_lines = []
        total_text_height = 0
        
        for font_size in range(40, 9, -2):
            font = ImageFont.truetype(self.font_path, font_size)
            lines = self.wrap_text(text, font, max_text_width)
            if not lines: continue
            line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
            total_height = sum(line_heights) + (len(lines) - 1) * 4
            if total_height <= max_text_height:
                best_font_size = font_size
                best_lines = lines
                total_text_height = total_height
                break
                
        if not best_lines:
            font = ImageFont.truetype(self.font_path, 10)
            best_lines = self.wrap_text(text, font, max_text_width)
            total_text_height = sum([font.getbbox(l)[3] - font.getbbox(l)[1] for l in best_lines])
            best_font_size = 10
            
        return best_font_size, best_lines, total_text_height

    def draw_text_on_bubble(self, cleaned_bubble_img, translated_text):
        bubble_h, bubble_w, _ = cleaned_bubble_img.shape
        pil_img = Image.fromarray(cv2.cvtColor(cleaned_bubble_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        optimal_size, lines, total_height = self.optimize_layout(translated_text, bubble_w, bubble_h)
        font = ImageFont.truetype(self.font_path, optimal_size)
        
        current_y = (bubble_h - total_height) // 2
        
        for line in lines:
            line_width = font.getbbox(line)[2] - font.getbbox(line)[0]
            line_height = font.getbbox(line)[3] - font.getbbox(line)[1]
            current_x = (bubble_w - line_width) // 2
            draw.text((current_x, current_y), line, font=font, fill=(0, 0, 0))
            current_y += line_height + 4
            
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def process_pipeline(self, input_bgr_img):
        is_valid, message = self.check_constraints(input_bgr_img)
        if not is_valid:
            raise ui.Error(message)

        output_img = input_bgr_img.copy()
        bubbles = self.detect_speech_bubbles(output_img)
        
        for bubble in bubbles:
            x, y, w, h = bubble["box"]
            cropped_bubble = input_bgr_img[y:y+h, x:x+w]
            
            try:
                pil_crop = Image.fromarray(cv2.cvtColor(cropped_bubble, cv2.COLOR_BGR2RGB))
                japanese_text = self.mocr(pil_crop)
                
                if len(japanese_text.strip()) < 2: 
                    continue 
                
                korean_text = self.translate_with_llm(japanese_text)
                print(f"🎯 인식 성공: {japanese_text}  -->  번역: {korean_text}")
                
                cleaned_bubble = self.clean_japanese_text(cropped_bubble)
                typeset_bubble = self.draw_text_on_bubble(cleaned_bubble, korean_text)
                
                output_img[y:y+h, x:x+w] = typeset_bubble
            except Exception as e:
                continue
                
        height, width = output_img.shape[:2]
        final_img = cv2.resize(output_img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        return final_img

# --- Gradio UI ---
engine = RealMangaTranslatorV27()

def predict(img):
    if img is None: return None
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    res_bgr = engine.process_pipeline(bgr)
    return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB)

demo = ui.Interface(
    fn=predict,
    inputs=ui.Image(type="pil", label="원본 이미지"),
    outputs=ui.Image(label="결과 이미지"),
    title="AI 만화 번역기 v2.7 (세로형 말풍선 종횡비 최적화)"
)
demo.launch(debug=True)
