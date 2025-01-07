import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import glob
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QWidget, QFileDialog, QLabel, QLineEdit,
                             QSpinBox, QScrollArea, QListWidget, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os


class FontComparisonTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("폰트 비교 도구")
        self.setGeometry(100, 100, 1200, 800)

        # 상태 변수 초기화
        self.font_directory = ""
        self.current_fonts = []  # 빈 리스트로 초기화
        self.current_index = -1  # -1로 초기화

        self.init_ui()

    def init_ui(self):
        # 메인 위젯과 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)  # 전체를 수평 레이아웃으로 변경

        # 왼쪽 패널 (폰트 리스트)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 폰트 리스트 위젯
        self.font_list = QListWidget()
        self.font_list.currentRowChanged.connect(self.on_list_selection_changed)
        left_layout.addWidget(QLabel("폰트 목록"))
        left_layout.addWidget(self.font_list)


        # 삭제 버튼 추가
        delete_layout = QHBoxLayout()
        self.delete_btn = QPushButton("선택한 폰트 삭제")
        self.delete_btn.clicked.connect(self.delete_font)
        self.delete_btn.setStyleSheet("background-color: #ff6b6b; color: white;")
        delete_layout.addWidget(self.delete_btn)
        left_layout.addLayout(delete_layout)

        # 오른쪽 패널
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 상단 컨트롤 영역
        control_layout = QHBoxLayout()

        # 폴더 선택 버튼
        self.folder_btn = QPushButton("폰트 폴더 선택")
        self.folder_btn.clicked.connect(self.select_folder)
        control_layout.addWidget(self.folder_btn)

        # 폴더 경로 표시
        self.folder_label = QLabel("선택된 폴더: ")
        control_layout.addWidget(self.folder_label)

        # 텍스트 입력 영역
        '''
        이유는 다음과 같습니다:
        네, 동의합니다. 해당 문자들은 시각적으로 강한 특징과 독특한 구조를 가지고 있어, 다양한 시각적 변주를 제공하기에 적합합니다. 아래와 같은 이유로 각 문자가 특이성이 있다고 볼 수 있습니다:
        
        Q, Z, X, K, O:
        
        Q: O에 꼬리가 더해져 독특한 형태를 만듭니다.
        Z: 지그재그로 명확한 각을 가집니다.
        X: 대칭적인 교차 구조로 눈에 띕니다.
        K: 수직봉과 대칭적인 대각선이 독특합니다.
        O: 완벽한 원으로 다른 문자들과 대비됩니다.
        g, j, y (하단 확장 문자들):
        
        g: 동그라미와 아래쪽 꼬리로 개성이 있습니다.
        j: 단순한 점과 아래로 길게 확장된 꼬리로 차별화됩니다.
        y: 하단으로 확장되며 V자와 유사한 모양이 독특합니다.
        W, M (추가적으로 대칭적이거나 폭이 넓은 문자들):
        
        W: 넓고 균형 잡힌 지그재그 형태로 공간감을 줍니다.
        M: 대칭성과 뾰족한 꼭짓점이 특징적입니다.
        p, b (반대 방향의 수직봉과 원의 결합):
        
        p: 상단에 원이 붙은 구조.
        b: 하단에 원이 붙은 구조로 p와 반대 방향.
        t:
        
        십자가 형태가 눈에 잘 띄며, 단순하지만 강렬한 인상을 줍니다.
        이 문자들을 선택하면 구조적, 대칭적, 확장적인 특징들이 혼합되어 더 다양한 시각적 실험과 활용이 가능할 것 같습니다.
        '''
        text_layout = QHBoxLayout()
        self.korean_input = QLineEdit("다람쥐")
        self.korean_input.setPlaceholderText("한글 텍스트 입력")
        self.english_input = QLineEdit("QZXKgfjyOWMpbt")
        self.english_input.setPlaceholderText("영문 텍스트 입력")
        text_layout.addWidget(QLabel("한글:"))
        text_layout.addWidget(self.korean_input)
        text_layout.addWidget(QLabel("영문:"))
        text_layout.addWidget(self.english_input)

        # 폰트 크기 설정
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 72)
        self.font_size_spin.setValue(60)
        text_layout.addWidget(QLabel("폰트 크기:"))
        text_layout.addWidget(self.font_size_spin)

        # 분석 시작 버튼
        self.analyze_btn = QPushButton("폰트 분석 시작")
        self.analyze_btn.clicked.connect(self.analyze_fonts)
        text_layout.addWidget(self.analyze_btn)

        # 네비게이션 버튼
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("이전 폰트")
        self.prev_btn.clicked.connect(self.show_previous)
        self.next_btn = QPushButton("다음 폰트")
        self.next_btn.clicked.connect(self.show_next)
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.next_btn)

        # 현재 폰트 정보 표시
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)

        # Matplotlib Figure 생성
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.canvas = FigureCanvas(self.figure)

        # 스크롤 영역에 캔버스 추가
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)

        # 오른쪽 패널에 위젯 추가
        right_layout.addLayout(control_layout)
        right_layout.addLayout(text_layout)
        right_layout.addLayout(nav_layout)
        right_layout.addWidget(self.info_label)
        right_layout.addWidget(scroll)

        # 메인 레이아웃에 左右 패널 추가
        main_layout.addWidget(left_panel, 1)  # stretch factor 1
        main_layout.addWidget(right_panel, 3)  # stretch factor 3

        # 초기 버튼 상태 설정
        self.update_button_states()

    def delete_font(self):
        current_item = self.font_list.currentItem()
        if not current_item:
            return

        font_path = str(Path(self.font_directory) / current_item.text())

        # 확인 대화상자 표시
        reply = QMessageBox.question(
            self,
            '폰트 삭제 확인',
            f'정말로 다음 폰트를 삭제하시겠습니까?\n{current_item.text()}',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # 파일 삭제
                os.remove(font_path)

                # 리스트에서 항목 제거
                row = self.font_list.currentRow()
                self.font_list.takeItem(row)

                # 성공 메시지
                QMessageBox.information(
                    self,
                    '삭제 완료',
                    '폰트 파일이 성공적으로 삭제되었습니다.',
                    QMessageBox.Ok
                )

                # 다음 폰트가 있으면 선택
                if self.font_list.count() > 0:
                    if row >= self.font_list.count():
                        row = self.font_list.count() - 1
                    self.font_list.setCurrentRow(row)
                    self.analyze_fonts()
                else:
                    # 모든 폰트가 삭제된 경우
                    self.info_label.setText("폰트가 없습니다.")
                    self.ax1.clear()
                    self.ax2.clear()
                    self.canvas.draw()

            except Exception as e:
                QMessageBox.critical(
                    self,
                    '삭제 실패',
                    f'폰트 파일 삭제 중 오류가 발생했습니다:\n{str(e)}',
                    QMessageBox.Ok
                )


    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "폰트 폴더 선택")
        if folder:
            self.font_directory = folder
            self.folder_label.setText(f"선택된 폴더: {folder}")

            # 폰트 파일 찾기
            font_files = []
            font_files.extend(glob.glob(str(Path(folder) / "*")))
            font_files.extend(glob.glob(str(Path(folder) / "*")))

            # 리스트 위젯 업데이트
            self.font_list.clear()
            for font_path in font_files:
                self.font_list.addItem(Path(font_path).name)

            # 첫 번째 폰트 선택
            if self.font_list.count() > 0:
                self.font_list.setCurrentRow(0)
                self.analyze_fonts()

    def analyze_fonts(self):
        # 선택된 폰트만 분석
        current_item = self.font_list.currentItem()
        if not current_item:
            self.info_label.setText("선택된 폰트가 없습니다.")
            return

        font_path = str(Path(self.font_directory) / current_item.text())
        try:
            korean_result = self.analyze_single_font(
                font_path,
                self.korean_input.text(),
                self.font_size_spin.value()
            )
            english_result = self.analyze_single_font(
                font_path,
                self.english_input.text(),
                self.font_size_spin.value()
            )

            if korean_result and english_result:
                similarity = 1 - (
                        abs(korean_result['corner_density'] - english_result['corner_density']) +
                        abs(korean_result['corner_std'] - english_result['corner_std'])
                ) / 2

                result = {
                    'font_path': font_path,
                    'font_name': Path(font_path).stem,
                    'similarity_score': similarity,
                    'korean_analysis': korean_result,
                    'english_analysis': english_result
                }

                # 분석 결과 표시
                self.show_analysis_result(result)

        except Exception as e:
            self.info_label.setText(f"폰트 분석 중 오류 발생: {str(e)}")

    def show_analysis_result(self, result):
        # 정보 업데이트
        self.info_label.setText(
            f"폰트: {result['font_name']}\n"
            f"유사도 점수: {result['similarity_score']:.3f}\n"
            f"파일: {result['font_path']}"
        )

        # 그래프 업데이트 (2x2 서브플롯으로 변경)
        self.ax1.clear()
        self.ax2.clear()

        # 왼쪽 위: 한글 원본
        self.ax1.imshow(result['korean_analysis']['image'], cmap='gray')
        self.ax1.set_title(f'한글 텍스트: {self.korean_input.text()}')
        self.ax1.axis('off')

        # 오른쪽 위: 영문 원본
        self.ax2.imshow(result['english_analysis']['image'], cmap='gray')
        self.ax2.set_title(f'영문 텍스트: {self.english_input.text()}')
        self.ax2.axis('off')

        self.figure.suptitle(
            f'폰트: {result["font_name"]}\n유사도 점수: {result["similarity_score"]:.3f}'
        )
        self.canvas.draw()
    def analyze_single_font(self, font_path, text, font_size):
        img = self.create_text_image(text, font_path, font_size)
        if img is None:
            return None

        # 코너 검출을 위해 그레이스케일 이미지 사용
        corners = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
        corners = cv2.normalize(corners, None, 0, 1, cv2.NORM_MINMAX)

        return {
            'image': img,  # 원본 이미지 저장
            'corners': corners,
            'corner_count': np.sum(corners > 0.01),
            'corner_density': np.mean(corners),
            'corner_std': np.std(corners)
        }

    def create_text_image(self, text, font_path, font_size, size=(500, 100)):
        try:
            img = Image.new('L', size, color=255)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, font_size)

            # 텍스트 중앙 정렬
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = (size[0] - text_width) // 2
            y = (size[1] - text_height) // 2

            draw.text((x, y), text, font=font, fill=0)
            return np.array(img)
        except Exception as e:
            print(f"Error creating image for font {font_path}: {str(e)}")
            return None

    def show_current_font(self):
        if not self.current_fonts or self.current_index < 0:
            return

        result = self.current_fonts[self.current_index]

        # 정보 업데이트
        self.info_label.setText(
            f"폰트: {result['font_name']}\n"
            f"유사도 점수: {result['similarity_score']:.3f}\n"
            f"파일: {result['font_path']}"
        )

        # 그래프 업데이트
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.imshow(result['korean_analysis']['corners'], cmap='hot')
        self.ax1.set_title(f'한글 텍스트\n코너 포인트: {result["korean_analysis"]["corner_count"]:.0f}')
        self.ax1.axis('off')

        self.ax2.imshow(result['english_analysis']['corners'], cmap='hot')
        self.ax2.set_title(f'영문 텍스트\n코너 포인트: {result["english_analysis"]["corner_count"]:.0f}')
        self.ax2.axis('off')

        self.figure.suptitle(
            f'폰트: {result["font_name"]}\n유사도 점수: {result["similarity_score"]:.3f}'
        )
        self.canvas.draw()

    def show_previous(self):
        current_row = self.font_list.currentRow()
        if current_row > 0:
            self.font_list.setCurrentRow(current_row - 1)
            self.analyze_fonts()

    def show_next(self):
        current_row = self.font_list.currentRow()
        if current_row < self.font_list.count() - 1:
            self.font_list.setCurrentRow(current_row + 1)
            self.analyze_fonts()

    def on_list_selection_changed(self):
        if self.font_list.currentRow() >= 0:
            self.analyze_fonts()
            self.update_button_states()

    def update_button_states(self):
        current_row = self.font_list.currentRow()
        # 이전 버튼은 현재 행이 0보다 클 때만 활성화
        self.prev_btn.setEnabled(current_row > 0)

        # 다음 버튼은 현재 행이 마지막 행보다 작을 때만 활성화
        self.next_btn.setEnabled(current_row < self.font_list.count() - 1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FontComparisonTool()
    window.show()
    sys.exit(app.exec_())