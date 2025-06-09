from loading_datasets import datasets_path, speed_1_lst  # 可改为 speed_2_lst 或 speed_3_lst

import os
import re

import cv2
import numpy as np  # 导入 numpy


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
    使用 OpenCV 显示图像序列，应用 HSV 模板，并结合形态学去噪和连通分量分析。
    同时会追踪并返回每个图像中保留下来的连通分量的信息。
    """
    # 定义 ROI 区域
    roi_x_min = 820
    roi_x_max = 1360
    roi_y_min = 100
    roi_y_max = 980

    # 定义 HSV 模板
    lower_hsv = np.array([10, 0, 30])
    upper_hsv = np.array([30, 255, 255])

    # 定义形态学操作的核
    # 3x3 或 5x5 的核通常效果不错，可以根据实际情况调整
    morph_kernel = np.ones((5, 5), np.uint8)

    # 定义面积阈值，所有小于此面积的连通分量将被视为噪声并移除
    min_noise_area_threshold = 1000

    # 存储每一帧的连通分量信息，如果你不需要返回这些信息，可以移除此变量
    all_frames_components_info = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Failed to load: {path}")
            continue

        # 1. 创建一个全白的背景图像
        processed_img = np.full(img.shape, 255, dtype=np.uint8)

        # 2. 将 ROI 区域内的原始图像内容复制到 processed_img
        h, w, _ = img.shape
        roi_y_min_clamped = max(0, roi_y_min)
        roi_y_max_clamped = min(h, roi_y_max)
        roi_x_min_clamped = max(0, roi_x_min)
        roi_x_max_clamped = min(w, roi_x_max)

        if (roi_y_max_clamped > roi_y_min_clamped) and \
                (roi_x_max_clamped > roi_x_min_clamped):
            processed_img[roi_y_min_clamped:roi_y_max_clamped, \
            roi_x_min_clamped:roi_x_max_clamped] = \
                img[roi_y_min_clamped:roi_y_max_clamped, \
                roi_x_min_clamped:roi_x_max_clamped].copy()

        current_frame_components_info = []  # 存储当前帧的连通分量信息

        # 3. 在 ROI 区域内应用 HSV 过滤，并进行去噪
        roi_img_for_processing = processed_img[roi_y_min_clamped:roi_y_max_clamped, \
                                 roi_x_min_clamped:roi_x_max_clamped]

        if roi_img_for_processing.size > 0:
            hsv = cv2.cvtColor(roi_img_for_processing, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

            # --- 3.1 先进行形态学开运算去噪 ---
            # 有助于去除小亮点和连接断裂的区域，平滑边缘
            denoised_mask_morph = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)

            # --- 3.2 再进行连通分量分析和面积过滤 ---
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(denoised_mask_morph, connectivity=8)

            # 创建一个用于最终显示的干净掩码
            final_display_mask = np.zeros_like(denoised_mask_morph, dtype=np.uint8)

            # 遍历所有连通分量（从标签 1 开始，标签 0 通常是背景）
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_noise_area_threshold:
                    # 保留符合面积要求的连通分量
                    final_display_mask[labels == i] = 255

                    # --- 追踪连通分量信息 ---
                    # 存储每个保留下来的连通分量的详细信息
                    # 可以根据需要添加更多信息，例如轮廓等
                    component_info = {
                        'label': i,
                        'area': area,
                        'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                                 stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                        'centroid': (centroids[i, 0], centroids[i, 1])
                    }
                    current_frame_components_info.append(component_info)

            all_frames_components_info.append(current_frame_components_info)  # 存储当前帧所有连通分量信息

            # 4. 根据 final_display_mask 更新 processed_img
            # 将 ROI 区域内，在 final_display_mask 中为 0 的像素设置为白色（背景色）
            processed_img[roi_y_min_clamped:roi_y_max_clamped, \
            roi_x_min_clamped:roi_x_max_clamped][final_display_mask == 0] = 255

            # --- 可选：在图像上绘制连通分量信息，便于调试 ---
            for comp in current_frame_components_info:
                x, y, w_comp, h_comp = comp['bbox']
                center_x, center_y = int(comp['centroid'][0]), int(comp['centroid'][1])
                # 绘制矩形框
                cv2.rectangle(processed_img[roi_y_min_clamped:roi_y_max_clamped, \
                                            roi_x_min_clamped:roi_x_max_clamped], \
                              (x, y), (x + w_comp, y + h_comp), (0, 255, 0), 2) # 绿色框
                # 绘制中心点
                cv2.circle(processed_img[roi_y_min_clamped:roi_y_max_clamped, \
                                          roi_x_min_clamped:roi_x_max_clamped], \
                           (center_x, center_y), 5, (0, 0, 255), -1) # 红色点

        # 缩放图像到目标窗口大小
        resized = cv2.resize(processed_img, window_size)

        cv2.imshow("Color Frame", resized)
        key = cv2.waitKey(50)  # 50ms 显示每帧
        if key == 27:  # 按 ESC 退出
            break
    cv2.destroyAllWindows()
    return all_frames_components_info  # 返回所有帧的连通分量信息


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
        # 调用 visualize_images 并获取返回的连通分量信息
        components_data = visualize_images(image_list)
        # 此时 components_data 包含了每一帧所有符合条件的连通分量信息
        # 你可以在这里对 components_data 进行进一步的分析或保存
        print(f"Finished processing folder: {folder}. Collected components data for {len(components_data)} frames.")
        # print("Example data for first frame:", components_data[0] if components_data else "No data")


if __name__ == "__main__":
    main()