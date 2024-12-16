from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
import os


@dataclass
class FontImageGenerator:
    image_size: Tuple[int, int] = (512, 512)  # 이미지 크기 지정
    output_dir: str = "datasets/font_dataset"  # 출력 폴더 지정

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
            image_path = os.path.join(self.output_dir, f"image{index}.png")
            txt_path = os.path.join(self.output_dir, f"image{index}.txt")

            image.save(image_path)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"sandoll style, bold serif font, sharp edges, high contrast strokes, professional typeface, elegant serif terminals, modern classic style, clean letterform, black text on white background")

        except Exception as e:
            print(f"폰트 '{font_path}' 처리 중 오류 발생: {str(e)}")


def main():
    # 이미지 생성기 인스턴스 생성
    generator = FontImageGenerator()

    # 영문자 배열 생성
    arr_eng = []
    for i in range(65, 91):  # A-Z
        arr_eng.append(chr(i))
    for i in range(97, 123):  # a-z
        arr_eng.append(chr(i))

    # 각 문자에 대해 이미지 생성
    for idx, char in enumerate(arr_eng, start=1):
        generator.create_font_image(char, "/Users/doheyonkim/data/fontbox/ttfs/en-20/0a1d7b390e726f4183a934210cfeee22", idx)  # 폰트 경로는 실제 경로로 수정 필요


if __name__ == "__main__":
    main()