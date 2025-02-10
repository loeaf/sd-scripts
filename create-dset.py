import argparse
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

    def create_font_image(self, text: str, font_path: str, index: int, trainPath: str) -> None:
        """텍스트를 이미지로 변환하고 저장"""
        self.output_dir = trainPath
        os.makedirs(self.output_dir, exist_ok=True)
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

    def create_sumnail_image(self, text: str, font_path: str, file_name: str, sumnailPath: str) -> None:
        """텍스트를 이미지로 변환하고 저장"""
        image = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(image)
        self.sumnail_dir = sumnailPath
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--font_pairs_path', type=str, default='/Users/doheyonkim/Depot/sd-scripts/font_pairs.csv', help='Path to CSV file containing dataset paths')
    parser.add_argument('--type', type=str, default='en', help='Path to CSV file containing dataset paths')
    args = parser.parse_args()
    font_pairs_path = args.font_pairs_path

    files = []
    with open(font_pairs_path, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            fontPath = row[0]
            uuid = row[2]
            trainPath = row[3]
            sumnailPath = row[5]
            # if ko or en
            if args.type == 'ko':
                # 하얀 뭉게구름 속 노을 빛 활자
                arr = ['하', '얀', '뭉', '게', '구', '름', '속', '노', '을', '빛', '활', '자']
            elif args.type == 'cz':
                # 今国意我永然警酬随
                arr = ['今', '国', '意', '我', '永', '然', '警', '酬', '随']
            else:
                # arr = ['Q', 'Z', 'X', 'K', 'g', 'f', 'j', 'y', 'O', 'W', 'M', 'p', 'b', 't']
                # H O C T X
                arr = ['H', 'O', 'C', 'T', 'X']
            # 각 문자에 대해 이미지 생성
            generator = FontImageGenerator()
            for idx, char in enumerate(arr, start=1):
                generator.create_font_image(char, fontPath, idx, trainPath)  # 폰트 경로는 실제 경로로 수정 필요
            if args.type == 'ko':
                generator.create_sumnail_image('하얀 뭉게구름 속 노을 빛 활자', fontPath, uuid, sumnailPath)
            elif args.type == 'cz':
                generator.create_sumnail_image('今国意我永然警酬随', fontPath, uuid, sumnailPath)
            else:
                generator.create_sumnail_image('H O C T X', fontPath, uuid, sumnailPath)


if __name__ == "__main__":
    main()