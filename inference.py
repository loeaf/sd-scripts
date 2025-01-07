import glob

import requests
import base64
from PIL import Image
import io
import json
import os
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
import os
import numpy as np


class StableDiffusionAPI:
    def __init__(self, url="http://127.0.0.1:7861"):
        self.url = url

    def encode_image_to_base64(self, image_path):
        with Image.open(image_path) as img:
            # Convert to RGB if image is in RGBA format
            if img.mode == 'RGBA':
                img = img.convert('RGB')

            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def img2img(self,
                image_path,
                prompt,
                negative_prompt="",
                checkpoint="",  # SD 모델 체크포인트
                vae="auto",  # VAE 설정
                lora_list=None,  # LoRA 리스트
                steps=20,
                cfg_scale=7,
                denoising_strength=0.75,
                batch_size=1):

        # 이미지를 base64로 인코딩
        encoded_image = self.encode_image_to_base64(image_path)

        # LoRA 프롬프트 처리
        if lora_list:
            for lora in lora_list:
                prompt += f"<lora:{lora['name']}:{lora['weight']}>"

        # API 요청 페이로드
        payload = {
            "init_images": [encoded_image],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "denoising_strength": denoising_strength,
            "batch_size": batch_size,
            "sampler_name": "DPM++ 2M SDE",
            "sampler_index": "DPM++ 2M SDE",
            "scheduler": "Karras",
            "override_settings": {
                "sd_model_checkpoint": checkpoint,
                "sd_vae": vae,
            }
        }

        try:
            response = requests.post(url=f"{self.url}/sdapi/v1/img2img",
                                     json=payload)
            response.raise_for_status()
            r = response.json()

            # 결과 이미지 저장
            for i, img_base64 in enumerate(r['images']):
                image = Image.open(io.BytesIO(base64.b64decode(img_base64)))

                # 결과 저장할 디렉토리 생성
                os.makedirs('outputs', exist_ok=True)

                # 파일명 생성 (원본 파일명 + 인덱스)
                original_filename = os.path.splitext(os.path.basename(image_path))[0]
                save_path = f"outputs/{original_filename}_output_{i}.png"

                image.save(save_path)
                print(f"이미지가 저장되었습니다: {save_path}")

            return r

        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return None

    def process_and_merge_outputs(self, output_dir: str = 'outputs', target_size: Tuple[int, int] = (128, 128), denoise_str: str = '') -> str:
        """
        출력 디렉토리의 이미지들을 처리하고 하나의 이미지로 병합

        Args:
            output_dir: 출력 이미지들이 있는 디렉토리
            target_size: 각 이미지의 목표 크기

        Returns:
            str: 병합된 이미지의 저장 경로
        """
        # 출력 디렉토리의 모든 PNG 파일 가져오기
        output_files = sorted(glob.glob(os.path.join(output_dir, '*_output_*.png')))

        if not output_files:
            print("처리할 이미지가 없습니다.")
            return None

        # 이미지 로드 및 리사이징
        images = []
        for file_path in output_files:
            with Image.open(file_path) as img:
                # 이미지 리사이징
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                images.append(resized_img)

        # 최종 이미지 크기 계산
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        # 새 이미지 생성
        merged_image = Image.new('RGB', (total_width, max_height))

        # 이미지 붙이기
        x_offset = 0
        for img in images:
            merged_image.paste(img, (x_offset, 0))
            x_offset += img.width

        # 결과 저장
        merged_path = os.path.join(output_dir, f'merged_output_{denoise_str}.png')
        merged_image.save(merged_path)
        print(f"병합된 이미지가 저장되었습니다: {merged_path}")
        # 이전 파일 삭제
        for file_path in output_files:
            os.remove(file_path)
        return merged_path


@dataclass
class FontImageGenerator:
    image_size: Tuple[int, int] = (512, 512)  # 이미지 크기 지정
    output_dir: str = "datasets/font_dataset2"  # 출력 폴더 지정

    def __post_init__(self):
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)

    def create_font_image(self, text: str, font_path: str, index: int) -> None:
        """텍스트를 이미지로 변환하고 저장"""
        image = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(image)

        try:
            # 폰트 크기 조정
            font_size = 1
            font = ImageFont.truetype(font_path, font_size)

            while True:
                next_font = ImageFont.truetype(font_path, font_size + 1)
                bbox = draw.textbbox((0, 0), text, font=next_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                if text_width > self.image_size[0] * 0.8 or text_height > self.image_size[1] * 0.8:
                    break

                font = next_font
                font_size += 1

            # 텍스트 중앙 정렬
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((self.image_size[0] - text_width) // 2 - bbox[0],
                        (self.image_size[1] - text_height) // 2 - bbox[1])

            # 텍스트 그리기
            draw.text(position, text, font=font, fill='black')

            # 이미지와 텍스트 파일 저장
            image_path = os.path.join(self.output_dir, f"{index}_{text}.png")
            image.save(image_path)

            return image_path
        except Exception as e:
            print(f"폰트 '{font_path}' 처리 중 오류 발생: {str(e)}")


# 사용 예시
if __name__ == "__main__":
    sd_api = StableDiffusionAPI()

    # LoRA 설정 예시
    # /sdapi/v1/loras 에서 사용 가능한 LoRA 목록 확인
    raw_lora = requests.get("http://localhost:7861/sdapi/v1/loras").json()
    # lambda name raw_lora
    loras = []
    for lora in raw_lora:
        loras.append({"name": lora['name'], "weight": 1})

    target_font = "0a9e276089efae01cf502c81eb902f00"
    # lambda filter aaa loras
    lora = list(filter(lambda x: x['name'] == target_font, loras))

    generator = FontImageGenerator()
    data = ['뭉', '게', '구', '름', '노', '을', '빛', '하', '늘']
    for denoise_str in np.arange(0.3, 0.7, 0.1):
        for index, data_char in enumerate(data):
            image_path = generator.create_font_image(data_char, "./481d2eaca08bef8b2e833324477d674d", index)
            # 이미지 변환 실행
            result = sd_api.img2img(
                image_path=image_path,
                prompt="masterpiece, (best quality:1.1), ",
                negative_prompt="",
                checkpoint="Anything-v4.5-pruned",  # 사용할 체크포인트
                vae="auto",
                lora_list=lora,
                steps=40,
                cfg_scale=7,
                denoising_strength=denoise_str
            )
        # 512x512 개수만큼의 여러개 이미지가 떨어지면 128x128로 샘플링 후 해당 이미지를 가로로 붙여서 하나의 이미지로 만들어서 저장
        merged_image_path = sd_api.process_and_merge_outputs(output_dir='outputs', target_size=(128, 128),
                                                             denoise_str=str(denoise_str))