from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
import os

import csv

@dataclass
class FontImageGenerator:
    image_size: Tuple[int, int] = (512, 512)  # 이미지 크기 지정
    output_dir: str = "datasets/font_dataset"  # 출력 폴더 지정
    sumnail_dir: str = "datasets/font_dataset/sumnail"  # 출력 폴더 지정

    def __post_init__(self):
        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.sumnail_dir, exist_ok=True)

    def set_output_dir(self, output_dir: str) -> None:
        self.output_dir = output_dir
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
            image_path = os.path.join(self.output_dir, f"image{index}.png")
            txt_path = os.path.join(self.output_dir, f"image{index}.txt")

            image.save(image_path)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"sandoll style, bold serif font, sharp edges, high contrast strokes, professional typeface, elegant serif terminals, modern classic style, clean letterform, black text on white background")

        except Exception as e:
            print(f"폰트 '{font_path}' 처리 중 오류 발생: {str(e)}")

    def create_sumnail_image(self, text: str, font_path: str, file_name: str) -> None:
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
            image_path = os.path.join(self.sumnail_dir, f"{file_name}.png")

            image.save(image_path)

        except Exception as e:
            print(f"폰트 '{font_path}' 처리 중 오류 발생: {str(e)}")


def list_files_in_directory(directory_path):
    try:
        # 디렉토리 내 파일 목록 가져오기
        files = os.listdir(directory_path)
        # . 숨김 파일 제외
        files = [file for file in files if not file.startswith('.')]
        return files
    except Exception as e:
        print(f"디렉토리 '{directory_path}' 처리 중 오류 발생: {str(e)}")
        return []

def main():
    # 예시 사용법
    # directory_path = '/Users/doheyonkim/data/fontbox/ttfs/en-20/'
    directory_path = '/home/user/resources/ttfs/en-20/'
    # save_path = '/Users/doheyonkim/Depot/sd-scripts/datasets/font_dataset'
    save_path = '/home/user/data/sd-scripts/datasets/font_dataset'
    # mkdir
    os.makedirs(save_path, exist_ok=True)
    files = list_files_in_directory(directory_path)
    csv_file_path = os.path.join(save_path, f"dataset.csv")

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Index', 'Data Path'])

        for idx, file in enumerate(files, start=1):
            # 영문자 배열 생성
            arr_eng = ['Q', 'Z', 'X', 'K', 'g', 'f', 'j', 'y', 'O', 'W', 'M', 'p', 'b', 't']
            # 각 문자에 대해 이미지 생성
            generator = FontImageGenerator()
            generator.set_output_dir(f"datasets/font_dataset/{file}")

            csv_writer.writerow([idx, f"datasets/font_dataset/{file}"])
            for idx, char in enumerate(arr_eng, start=1):
                generator.create_font_image(char, directory_path + file, idx)  # 폰트 경로는 실제 경로로 수정 필요
            generator.create_sumnail_image('QXKgfjyO', directory_path + file, file)



if __name__ == "__main__":
    main()