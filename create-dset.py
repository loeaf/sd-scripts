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
                # 가나더려모부쇼야져쵸켜튜프히
                # arr = ['하', '얀', '뭉', '게', '구', '름', '속', '노', '을', '빛', '활', '자']
                # arr = ['가', '나', '더', '려', '모', '부', '쇼', '야', '져', '쵸', '켜', '튜', '프', '히']
                arr = ['가', '나', '더', '려', '모', '부', '쇼', '야', '져', '쵸', '켜', '튜', '프', '히', '응',
                        '뭉', '게', '구', '름', '노', '을', '빛' , '짖', '속', '활', '자', '까', '싸', '따',
                       '쑥', '뿍', '갉', '밝', '꺍', '뱖', '쏾', '쒧', '앍', '악', '욱', '요', '앙', '양', '영', '융','영']
                # arr = ['다', '하', '지', '이', '기', '리', '가', '사', '자', '대', '적', '어', '아', '시', '장', '수', '되', '전', '상', '소', '부', '정', '나', '인', '일', '그', '주', '고', '도', '히', '구', '비', '치', '보', '제', '스', '오', '무', '생', '마', '신', '서', '연', '로', '내', '성', '학', '실', '화', '중', '공', '한', '국', '해', '관', '우', '여', '식', '문', '미', '용', '원', '의', '교', '방', '바', '간', '거', '음', '발', '모', '경', '조', '위', '저', '만', '개', '세', '요', '반', '물', '안', '르', '차', '외', '심', '분', '단', '통', '유', '선', '속', '계', '예', '과', '불', '금', '달', '입', '점', '동', '감', '출', '행', '산', '래', '진', '양', '회', '명', '재', '당', '려', '초', '체', '말', '러', '영', '건', '강', '라', '설', '집', '추', '작', '남', '각', '니', '피', '편', '매', '근', '터', '업', '버', '석', '들', '절', '결', '약', '직', '날', '손', '배', '복', '호', '표', '력', '품', '없', '색', '트', '활', '울', '새', '머', '살', '종', '청', '현', '운', '타', '판', '월', '면', '럽', '름', '참', '형', '확', '망', '야', '쪽', '임', '포', '민', '역', '목', '순', '별', '술', '급', '올', '두', '평', '년', '번', '질', '데', '담', '너', '늘', '천', '드', '극', '후', '파', '격', '증', '디', '필', '혼', '습', '노', '최', '창', '본', '접', '깨', '길', '프', '법', '루', '눈', '환', '은', '잠', '앞', '십', '코', '밤', '침', '난', '워', '꾸', '책', '열', '독', '철', '쓰', '군', '락', '잡', '벌', '움', '처', '합', '씨', '토', '향', '삼', '변', '능', '답', '육', '키', '님', '특', '먹', '갈', '송', '잘', '녀', '험', '료', '태']
            elif args.type == 'cz':
                # 今国意我永然警酬随
                arr = ['今', '国', '意', '我', '永', '然', '警', '酬', '随']
            else:
                arr = ['Z', 'X', 'K', 'y', 'O', 'W', 'M', 'p', 'b', 't']
                # H O C T X
                # arr = ['H', 'O', 'C', 'T', 'X']
            # 각 문자에 대해 이미지 생성
            generator = FontImageGenerator()
            for idx, char in enumerate(arr, start=1):
                generator.create_font_image(char, fontPath, idx, trainPath)  # 폰트 경로는 실제 경로로 수정 필요
            if args.type == 'ko':
                generator.create_sumnail_image('하얀 뭉게구름 속 노을 빛 활자', fontPath, uuid, sumnailPath)
            elif args.type == 'cz':
                generator.create_sumnail_image('今国意我永然警酬随', fontPath, uuid, sumnailPath)
            else:
                # arr to text
                arr2text = ' '.join(arr)
                generator.create_sumnail_image(arr2text, fontPath, uuid, sumnailPath)


if __name__ == "__main__":
    main()