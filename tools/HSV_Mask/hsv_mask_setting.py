"""
    HSVFilterApp - 多区间 HSV 阈值可视化调节工具（支持拖拽、自定义多模板）

    目前已经测试的最佳Mask：Range 1: Lower=[10, 0, 30], Upper=[30, 255, 255]

功能：
- 支持通过滑块交互调节 HSV 上下限
- 可添加多个 HSV 区间，基于“或”关系融合掩码
- 支持拖拽或按钮加载图像
- 实时显示原图与掩码图（掩码背景为青蓝色）
- 每个区间支持导出当前 HSV 配置文本
- 图像区域和参数区采用美观布局，跨平台兼容

依赖：
- PyQt5
- OpenCV (opencv-python)
- Python 3.6+

运行：
$ python hsv_mask_setting.py

作者：Jiaming Zhang
日期：2025-06
"""


import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QSlider, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QGroupBox, QScrollArea
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


def cv2qt(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)


class DualSlider(QWidget):
    """双滑块控件，共享一组标签，用于 H/S/V 的 min/max 设置"""
    def __init__(self, name, max_value, on_update):
        super().__init__()
        self.name = name
        self.max_value = max_value
        self.on_update = on_update

        layout = QVBoxLayout()
        title = QLabel(f"{name}")
        self.min_slider = QSlider(Qt.Horizontal)
        self.max_slider = QSlider(Qt.Horizontal)

        for slider in [self.min_slider, self.max_slider]:
            slider.setMinimum(0)
            slider.setMaximum(max_value)
            slider.valueChanged.connect(self.update_callback)

        self.min_slider.setValue(0)
        self.max_slider.setValue(max_value)

        layout.addWidget(title)
        layout.addWidget(QLabel("Min"))
        layout.addWidget(self.min_slider)
        layout.addWidget(QLabel("Max"))
        layout.addWidget(self.max_slider)

        self.setLayout(layout)

    def update_callback(self):
        if self.min_slider.value() > self.max_slider.value():
            self.max_slider.setValue(self.min_slider.value())
        self.on_update()

    def get_range(self):
        return self.min_slider.value(), self.max_slider.value()


class HSVBlock(QWidget):
    """表示一个完整的 HSV 区间（3 组双滑块）"""
    def __init__(self, on_update):
        super().__init__()
        self.on_update = on_update
        layout = QVBoxLayout()

        self.h_slider = DualSlider("H", 179, on_update)
        self.s_slider = DualSlider("S", 255, on_update)
        self.v_slider = DualSlider("V", 255, on_update)

        layout.addWidget(self.h_slider)
        layout.addWidget(self.s_slider)
        layout.addWidget(self.v_slider)

        self.setLayout(layout)

    def get_hsv_range(self):
        h1, h2 = self.h_slider.get_range()
        s1, s2 = self.s_slider.get_range()
        v1, v2 = self.v_slider.get_range()
        return np.array([h1, s1, v1]), np.array([h2, s2, v2])


class HSVFilterApp(QWidget):
    def __init__(self):

        super().__init__()
        self.setWindowTitle("HSV 阈值调节工具")
        self.setMinimumSize(1400, 800)
        self.setAcceptDrops(True)

        self.image = None
        self.hsv_blocks = []

        # 图像区域
        self.original_label = QLabel("原图")
        self.masked_label = QLabel("掩码图")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.masked_label.setAlignment(Qt.AlignCenter)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.original_label)
        image_layout.addWidget(self.masked_label)

        # 侧边栏：控制区
        self.add_button = QPushButton("添加 HSV 区间")
        self.add_button.clicked.connect(self.add_hsv_block)

        self.load_button = QPushButton("加载图片")
        self.load_button.clicked.connect(self.load_image)

        self.hsv_area = QVBoxLayout()
        self.hsv_area.addWidget(self.load_button)
        self.hsv_area.addWidget(self.add_button)

        # 文本导出框
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFixedHeight(120)
        self.hsv_area.addWidget(QLabel("当前 HSV 配置："))
        self.hsv_area.addWidget(self.output_text)

        hsv_container = QWidget()
        hsv_container.setLayout(self.hsv_area)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(hsv_container)
        scroll.setFixedWidth(350)

        # 总体布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(scroll)
        main_layout.addLayout(image_layout)

        self.setLayout(main_layout)

    def add_hsv_block(self):
        block = HSVBlock(self.update_mask)
        self.hsv_blocks.append(block)
        self.hsv_area.insertWidget(len(self.hsv_area.children()) - 2, block)
        self.update_mask()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            self.image = cv2.imread(file_path)
            self.update_mask()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                self.image = cv2.imread(file_path)
                self.update_mask()
                break

    def update_mask(self):
        if self.image is None:
            return



        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        text_out = ""

        for idx, block in enumerate(self.hsv_blocks):
            lower, upper = block.get_hsv_range()
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = cv2.bitwise_or(mask_total, mask)
            text_out += f"Range {idx+1}: Lower={lower.tolist()}, Upper={upper.tolist()}\n"

        # 创建青蓝背景图
        bg_color = (255, 255, 0)  # 青蓝色 (BGR)
        background = np.full(self.image.shape, bg_color, dtype=np.uint8)

        # masked = cv2.bitwise_and(self.image, self.image, mask=mask_total)
        # 将图像中 mask 区域保留，其余为青蓝色背景
        masked = np.where(mask_total[:, :, np.newaxis] == 0, background, self.image)
        self.output_text.setText(text_out)

        fig_size = (1440, 810)

        self.original_label.setPixmap(QPixmap.fromImage(cv2qt(self.image)).scaled(fig_size[0], fig_size[1], Qt.KeepAspectRatio))
        self.masked_label.setPixmap(QPixmap.fromImage(cv2qt(masked)).scaled(fig_size[0], fig_size[1], Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = HSVFilterApp()
    win.show()
    sys.exit(app.exec_())
