from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple
import os


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
            # txt_path = os.path.join(self.output_dir, f"image{text}.txt")

            image.save(image_path)
            # with open(txt_path, 'w', encoding='utf-8') as f:
            #     f.write(f"English character {text}, font character")

        except Exception as e:
            print(f"폰트 '{font_path}' 처리 중 오류 발생: {str(e)}")


def main():
    # 이미지 생성기 인스턴스 생성
    generator = FontImageGenerator()

    # 영문자 배열 생성
    # arr_eng = []ㅌㅌ
    # for i in range(65, 91):  # A-Z
    #     arr_eng.append(chr(i))
    # for i in range(97, 123):  # a-z
    #     arr_eng.append(chr(i))
    # for 가
    # 각 문자에 대해 이미지 생성
    # generator.create_font_image('현', "./481d2eaca08bef8b2e833324477d674d", 0)  # 폰트 경로는 실제 경로로 수정 필요
    # 얇은거
    # 그를 쏙 빼닮은 너희와 숲길 꽃 찾기
    data = ['그','를','쏙','빼','닮','은','너','희','와','숲','길','꽃','찾','기']
    for index, data in enumerate(data):
        # generator.create_font_image(data, "./db96f6e4dac8a65e529775aaa3d4dc27", index)
        generator.create_font_image(data, "./481d2eaca08bef8b2e833324477d674d", index)


if __name__ == "__main__":
    main()