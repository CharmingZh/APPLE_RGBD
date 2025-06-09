from loading_datasets import datasets_path, speed_1_lst  # 可改为 speed_2_lst 或 speed_3_lst

import os
import re

import cv2

def get_sorted_png_files(folder):
    """
    获取目录中以 color_frame_ 开头、.png 结尾，按帧编号排序的完整文件路径
    """
    all_files = os.listdir(folder)
    png_files = [
        f for f in all_files if f.endswith('.png') and f.startswith('color_frame_')
    ]

    def extract_index(fname):
        match = re.search(r'color_frame_(\d+)_', fname)
        return int(match.group(1)) if match else -1

    sorted_files = sorted(png_files, key=extract_index)
    full_paths = [os.path.join(folder, f) for f in sorted_files]
    return full_paths

def visualize_images(image_paths, window_size=(1440, 810)):
    """
    使用 OpenCV 显示图像序列
    """
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Failed to load: {path}")
            continue

        # 缩放图像到目标窗口大小
        resized = cv2.resize(img, window_size)

        cv2.imshow("Color Frame", resized)
        key = cv2.waitKey(1)  # 200ms 显示每帧
        if key == 27:  # 按 ESC 退出
            break
    cv2.destroyAllWindows()

def main():
    # 你可以换成 speed_2_lst 或 speed_3_lst
    selected_list = speed_1_lst

    for idx in selected_list:
        folder = datasets_path[idx]
        print(f"Processing folder: {folder}")
        image_list = get_sorted_png_files(folder)
        if not image_list:
            print("[WARNING] No PNGs found.")
            continue
        visualize_images(image_list)

if __name__ == "__main__":
    main()