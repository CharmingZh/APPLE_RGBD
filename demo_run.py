from loading_datasets import datasets_path, speed_1_lst  # 可改为 speed_2_lst 或 speed_3_lst

import os
import re
import cv2
import numpy as np


def get_sorted_png_files(folder):
    """
    获取目录中以 color_frame_ 开头、.png 结尾，按帧编号排序的完整文件路径
    """
    all_files = os.listdir(folder)
    png_files = [
        f for f in all_files if f.endswith('.png') and f.startswith('color_frame_')  # CORRECTED LINE HERE
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
    实现跨帧连通分量颜色一致性追踪，并按要求显示累加计数器和自定义编号。
    追踪逻辑已根据“两列从下向上运行”的特性进行优化。
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
    morph_kernel = np.ones((5, 5), np.uint8)

    # 定义面积阈值，所有小于此面积的连通分量将被视为噪声并移除
    min_noise_area_threshold = 2750

    # --- 追踪相关的变量 ---
    tracked_objects = {}  # key: unique_id, value: {'centroid': (x,y), 'color': (B,G,R), 'missed_frames': int, 'assigned_number': int}
    next_unique_id = 0
    global_component_counter = 0  # 累加计数器，只在新对象出现时增加

    # 追踪参数优化
    avg_vertical_movement_per_frame = 20  # 假设每帧向上移动20像素，请根据实际情况调整
    max_distance_for_tracking = 75  # 增大匹配距离，因为有了预测

    # 用于处理短暂消失的追踪对象
    max_missed_frames = 5  # 对象可以消失的帧数，超过则删除追踪

    # --- 自定义编号相关参数 ---
    num_objects_per_column = 9  # 每列对象的最大数量

    # 预定义一组区分度高的颜色，循环使用
    COLORS = [
        (255, 0, 0),  # 蓝色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 红色
        (255, 255, 0),  # 青色
        (0, 255, 255),  # 黄色
        (255, 0, 255),  # 洋红色
        (128, 0, 0),  # 深蓝色
        (0, 128, 0),  # 深绿色
        (0, 0, 128),  # 深红色
        (128, 128, 0),  # 青灰色
        (0, 128, 128),  # 蓝绿色
        (128, 0, 128),  # 紫色
        (64, 64, 64),  # 灰色
        (192, 192, 192)  # 浅灰色
    ]
    color_index = 0

    all_frames_components_info = []  # 存储每一帧的连通分量信息

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] Failed to load: {path}")
            continue

        processed_img = np.full(img.shape, 255, dtype=np.uint8)

        h, w, _ = img.shape
        roi_y_min_clamped = max(0, roi_y_min)
        roi_y_max_clamped = min(h, roi_y_max)
        roi_x_min_clamped = max(0, roi_x_min)
        roi_x_max_clamped = min(w, roi_x_max)

        roi_img_for_processing = img[roi_y_min_clamped:roi_y_max_clamped, \
                                 roi_x_min_clamped:roi_x_max_clamped].copy()

        # 计算 ROI 内部的宽度中点，用于区分左右两列
        roi_center_x_local = (roi_x_max_clamped - roi_x_min_clamped) / 2

        current_frame_tracked_components_info = []

        if roi_img_for_processing.size > 0:
            hsv = cv2.cvtColor(roi_img_for_processing, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            denoised_mask_morph = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(denoised_mask_morph, connectivity=8)

            current_frame_detected_components = []

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_noise_area_threshold:
                    x, y, w_comp, h_comp = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                    center_x, center_y = int(centroids[i, 0]), int(centroids[i, 1])

                    current_frame_detected_components.append({
                        'label_id': i,
                        'centroid': (center_x, center_y),
                        'bbox': (x, y, w_comp, h_comp),
                        'area': area
                    })

            # --- 追踪逻辑：匹配和分配ID及颜色 ---
            matched_current_components_flags = [False] * len(current_frame_detected_components)

            next_frame_tracked_objects = {}

            # 1. 尝试匹配现有追踪对象
            for tracked_id, obj_data in list(tracked_objects.items()):
                predicted_centroid_y = obj_data['centroid'][1] - avg_vertical_movement_per_frame
                predicted_centroid = (obj_data['centroid'][0], predicted_centroid_y)

                min_dist = float('inf')
                best_match_idx = -1

                for idx, current_comp in enumerate(current_frame_detected_components):
                    if not matched_current_components_flags[idx]:
                        dist = np.linalg.norm(np.array(predicted_centroid) - np.array(current_comp['centroid']))

                        if dist < min_dist and dist <= max_distance_for_tracking:
                            best_match_idx = idx
                            min_dist = dist

                if best_match_idx != -1:
                    matched_comp = current_frame_detected_components[best_match_idx]
                    matched_comp['unique_id'] = tracked_id
                    matched_comp['color'] = obj_data['color']
                    matched_comp['assigned_number'] = obj_data['assigned_number']  # 保持原编号
                    matched_current_components_flags[best_match_idx] = True

                    next_frame_tracked_objects[tracked_id] = {
                        'centroid': matched_comp['centroid'],
                        'color': obj_data['color'],
                        'missed_frames': 0,
                        'assigned_number': obj_data['assigned_number']
                    }
                else:
                    obj_data['missed_frames'] += 1
                    if obj_data['missed_frames'] <= max_missed_frames:
                        next_frame_tracked_objects[tracked_id] = obj_data

            # 2. 处理未匹配的当前帧组件 (新对象)
            for idx, current_comp in enumerate(current_frame_detected_components):
                if not matched_current_components_flags[idx]:
                    current_comp['unique_id'] = next_unique_id
                    current_comp['color'] = COLORS[color_index % len(COLORS)]
                    color_index += 1
                    next_unique_id += 1
                    global_component_counter += 1

                    current_comp['assigned_number'] = -1  # Placeholder for now

                    next_frame_tracked_objects[current_comp['unique_id']] = {
                        'centroid': current_comp['centroid'],
                        'color': current_comp['color'],
                        'missed_frames': 0,
                        'assigned_number': -1  # Placeholder
                    }

            tracked_objects = next_frame_tracked_objects

            # --- Now, assign custom numbers to all objects present in this frame ---
            # Collect all objects that are being tracked (or just newly detected) and are not marked as missed
            current_frame_components_for_visual_and_numbering = []
            for uid, obj_data in tracked_objects.items():
                # We need the original detection info (bbox, label_id) for visualization
                # So, find the current detection corresponding to this uid
                found_original_detection = None
                for detected_comp in current_frame_detected_components:
                    if 'unique_id' in detected_comp and detected_comp['unique_id'] == uid:
                        found_original_detection = detected_comp
                        break

                if found_original_detection:  # Only consider objects that were actually detected in this frame
                    # Ensure obj_data has the most up-to-date centroid from the current frame detection
                    obj_data['centroid'] = found_original_detection['centroid']
                    # Use the properties from the tracked object for color and ID
                    found_original_detection['unique_id'] = uid
                    found_original_detection['color'] = obj_data['color']
                    # assigned_number will be updated below
                    found_original_detection['assigned_number'] = obj_data['assigned_number']

                    current_frame_components_for_visual_and_numbering.append(found_original_detection)

            # Sort these components to assign numbers based on column and vertical position
            current_frame_components_for_visual_and_numbering.sort(
                key=lambda c: (c['centroid'][0] > roi_center_x_local, -c['centroid'][1]))

            # Iterate through the sorted list and assign/update assigned_number
            for idx_in_sorted_list, current_comp_viz in enumerate(current_frame_components_for_visual_and_numbering):
                column_type = 0
                if current_comp_viz['centroid'][0] > roi_center_x_local:
                    column_type = 1

                # The rank in the column is its position in the sorted list *among its column peers* + 1
                rank_in_column = 0
                for prev_comp_viz in current_frame_components_for_visual_and_numbering[:idx_in_sorted_list]:
                    prev_col_type = 0
                    if prev_comp_viz['centroid'][0] > roi_center_x_local:
                        prev_col_type = 1
                    if prev_col_type == column_type:
                        rank_in_column += 1

                assigned_number = rank_in_column + 1
                if column_type == 1:  # If right column
                    assigned_number += num_objects_per_column

                current_comp_viz['assigned_number'] = assigned_number

                # Update the assigned_number in the persistent tracked_objects as well
                tracked_objects[current_comp_viz['unique_id']]['assigned_number'] = assigned_number

            # Store components info for the current frame for external use
            filtered_current_frame_info = []
            for comp in current_frame_components_for_visual_and_numbering:
                filtered_current_frame_info.append({
                    'unique_id': comp['unique_id'],
                    'area': comp['area'],
                    'bbox': comp['bbox'],
                    'centroid': comp['centroid'],
                    'color': comp['color'],
                    'assigned_number': comp['assigned_number']
                })
            all_frames_components_info.append(filtered_current_frame_info)

            # --- 可视化：在 processed_img 上绘制连通分量和文本 ---
            colored_output_roi = np.zeros((roi_y_max_clamped - roi_y_min_clamped,
                                           roi_x_max_clamped - roi_x_min_clamped, 3), dtype=np.uint8)

            for comp in current_frame_components_for_visual_and_numbering:  # Iterate the sorted list for drawing
                colored_output_roi[labels == comp['label_id']] = comp['color']

                text_to_display = str(comp['assigned_number'])  # Display the custom assigned number

                text_x = comp['bbox'][0] + comp['bbox'][2] + 5
                text_y = comp['bbox'][1] + int(comp['bbox'][3] / 2)

                if text_x + 50 > colored_output_roi.shape[1] and comp['bbox'][0] - 50 > 0:
                    text_x = comp['bbox'][0] - 50

                cv2.putText(colored_output_roi, text_to_display,
                            (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2, cv2.LINE_AA)  # Text color changed to yellow for better visibility

            processed_img[roi_y_min_clamped:roi_y_max_clamped, \
            roi_x_min_clamped:roi_x_max_clamped] = colored_output_roi

        # --- 在屏幕左上角显示累加的连通分量总数 ---
        text_total_components = f"Total Detected: {global_component_counter}"
        cv2.putText(processed_img, text_total_components, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        resized = cv2.resize(processed_img, window_size)

        cv2.imshow("Color Frame", resized)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
    return all_frames_components_info


def main():
    selected_list = speed_1_lst

    for idx in selected_list:
        folder = datasets_path[idx]
        print(f"Processing folder: {folder}")
        image_list = get_sorted_png_files(folder)
        if not image_list:
            print("[WARNING] No PNGs found.")
            continue
        components_data = visualize_images(image_list)
        print(f"Finished processing folder: {folder}. Collected components data for {len(components_data)} frames.")


if __name__ == "__main__":
    main()