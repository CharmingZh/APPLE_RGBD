from loading_datasets import datasets_path, speed_1_lst  # 可改为 speed_2_lst 或 speed_3_lst

import os
import re
import cv2
import numpy as np
import csv
# 引入scipy库来使用最高效的全局最优匹配算法（匈牙利算法）
# 如果你的环境中没有安装，请运行: pip install scipy
from scipy.optimize import linear_sum_assignment

# ===================================================================
# 新增依赖: 用于数据处理和可视化
# 确保已安装: pip install pandas matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# ===================================================================

# ===================================================================
# 全局开关: 设置为 True 来保存视频, 设置为 False 则不保存
SAVE_VIDEO = False


# ===================================================================

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


def calculate_geometric_properties(component_mask, bbox):
    """
    为单个连通分量计算几何属性
    """
    properties = {
        'aspect_ratio': 0.0,
        'circularity': 0.0,
        'pca_avg_axis_length': 0.0
    }

    # 1. 宽高比
    x, y, w, h = bbox
    if h > 0:
        properties['aspect_ratio'] = w / h

    # 2. 圆度
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = contours[0]
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            properties['circularity'] = (4 * np.pi * area) / (perimeter ** 2)

    # 3. PCA 和轴长
    points = np.argwhere(component_mask == 1)
    if len(points) > 5:  # PCA需要至少5个点才能稳定
        try:
            # BUG修复: 使用 cv2.PCACompute2 替代 cv2.PCACompute
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(np.float32(points[:, ::-1]), mean=None)
            # 轴长近似为 4 * sqrt(特征值), 代表约2个标准差
            axis_length1 = 4 * np.sqrt(eigenvalues[0, 0])
            axis_length2 = 4 * np.sqrt(eigenvalues[1, 0])
            properties['pca_avg_axis_length'] = (axis_length1 + axis_length2) / 2
        except cv2.error:
            # 对于非常退化的形状（如一条直线），PCA可能会失败
            pass

    return properties


def visualize_images(image_paths, window_size=(1440, 810), save_video=False, output_path="output.mp4"):
    """
    处理图像序列以追踪对象，可视化结果，并收集几何统计数据。
    """
    roi_x_min, roi_x_max, roi_y_min, roi_y_max = 820, 1360, 100, 980
    lower_hsv = np.array([10, 0, 30])
    upper_hsv = np.array([30, 255, 255])
    morph_kernel = np.ones((5, 5), np.uint8)
    min_noise_area_threshold = 2750

    tracked_objects = {}
    next_unique_id = 0
    global_component_counter = 0
    initial_vertical_movement = 20
    max_distance_for_tracking = 100
    max_missed_frames = 5

    num_objects_per_column = 9
    next_left_column_number = 1
    next_right_column_number = num_objects_per_column + 1

    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0),
              (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128), (64, 64, 64), (192, 192, 192)]
    color_index = 0

    all_stats_data = []

    video_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, window_size)
        if not video_writer.isOpened():
            print(f"[错误] 无法打开视频文件进行写入: {output_path}")
            save_video = False
        else:
            print(f"将开始录制视频到: {output_path}")

    for frame_idx, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        roi_img = img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        roi_center_x_local = (roi_x_max - roi_x_min) / 2

        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        denoised_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(denoised_mask, connectivity=8)

        current_detections = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_noise_area_threshold:
                bbox = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH],
                        stats[i, cv2.CC_STAT_HEIGHT])
                component_mask = (labels == i).astype(np.uint8)
                geo_properties = calculate_geometric_properties(component_mask, bbox)
                detection_data = {'centroid': centroids[i], 'bbox': bbox, 'area': area, 'label_id': i}
                detection_data.update(geo_properties)
                current_detections.append(detection_data)

        tracked_ids = list(tracked_objects.keys())
        new_tracked_objects = {}

        matched_detection_indices = set()

        if tracked_ids:
            predicted_centroids = [tracked_objects[tid]['centroid'] + tracked_objects[tid]['velocity'] for tid in
                                   tracked_ids]
            cost_matrix = np.full((len(predicted_centroids), len(current_detections)), 1e6)
            for t_idx, pred_cen in enumerate(predicted_centroids):
                for d_idx, det in enumerate(current_detections):
                    dist = np.linalg.norm(pred_cen - det['centroid'])
                    if dist < max_distance_for_tracking: cost_matrix[t_idx, d_idx] = dist

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for t_idx, d_idx in zip(row_ind, col_ind):
                if cost_matrix[t_idx, d_idx] < max_distance_for_tracking:
                    tid = tracked_ids[t_idx]
                    detection = current_detections[d_idx]
                    old_data = tracked_objects[tid]

                    new_velocity = detection['centroid'] - old_data['centroid']
                    updated_velocity = old_data['velocity'] * 0.5 + new_velocity * 0.5

                    new_obj_data = {
                        'centroid': detection['centroid'], 'color': old_data['color'], 'missed_frames': 0,
                        'assigned_number': old_data['assigned_number'], 'velocity': updated_velocity,
                        'unique_id': tid, 'bbox': detection['bbox'], 'label_id': detection['label_id']
                    }
                    new_tracked_objects[tid] = new_obj_data
                    matched_detection_indices.add(d_idx)

                    all_stats_data.append({
                        'frame': frame_idx, 'assigned_number': new_obj_data['assigned_number'],
                        'unique_id': new_obj_data['unique_id'], 'area': detection['area'],
                        'aspect_ratio': detection['aspect_ratio'], 'circularity': detection['circularity'],
                        'pca_avg_axis_length': detection['pca_avg_axis_length']
                    })

            unmatched_track_ids = set(tracked_ids) - {tracked_ids[t_idx] for t_idx, d_idx in zip(row_ind, col_ind) if
                                                      cost_matrix[t_idx, d_idx] < max_distance_for_tracking}
            for tid in unmatched_track_ids:
                obj = tracked_objects[tid]
                obj['missed_frames'] += 1
                if obj['missed_frames'] <= max_missed_frames:
                    obj['centroid'] += obj['velocity']
                    new_tracked_objects[tid] = obj

        unmatched_detection_indices = set(range(len(current_detections))) - matched_detection_indices
        for d_idx in unmatched_detection_indices:
            det = current_detections[d_idx]
            is_right = det['centroid'][0] > roi_center_x_local
            assigned_num = next_right_column_number if is_right else next_left_column_number
            if is_right:
                next_right_column_number += 1
            else:
                next_left_column_number += 1

            new_obj_data = {
                'centroid': det['centroid'], 'color': COLORS[color_index % len(COLORS)], 'missed_frames': 0,
                'assigned_number': assigned_num, 'velocity': np.array([0, -initial_vertical_movement]),
                'unique_id': next_unique_id, 'bbox': det['bbox'], 'label_id': det['label_id']
            }
            new_tracked_objects[next_unique_id] = new_obj_data

            all_stats_data.append({
                'frame': frame_idx, 'assigned_number': new_obj_data['assigned_number'],
                'unique_id': new_obj_data['unique_id'], 'area': det['area'],
                'aspect_ratio': det['aspect_ratio'], 'circularity': det['circularity'],
                'pca_avg_axis_length': det['pca_avg_axis_length']
            })

            next_unique_id += 1;
            color_index += 1;
            global_component_counter += 1

        tracked_objects = new_tracked_objects

        processed_img = np.full(img.shape, 255, dtype=np.uint8)
        colored_output_roi = np.zeros_like(roi_img)
        if tracked_objects:
            for obj in tracked_objects.values():
                if obj['missed_frames'] == 0:
                    colored_output_roi[labels == obj['label_id']] = obj['color']
                    (x, y, w, h) = obj['bbox']
                    text_to_display = str(obj['assigned_number'])
                    text_x, text_y = int(x + w + 5), int(y + h / 2)
                    if text_x + 50 > colored_output_roi.shape[1] and x - 50 > 0: text_x = int(x - 50)
                    cv2.putText(colored_output_roi, text_to_display, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 0), 2, cv2.LINE_AA)

        processed_img[roi_y_min:roi_y_max, roi_x_min:roi_x_max] = colored_output_roi
        text_total_components = f"Detected in Total: {global_component_counter}"
        cv2.putText(processed_img, text_total_components, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                    cv2.LINE_AA)
        resized = cv2.resize(processed_img, window_size)
        if save_video and video_writer is not None: video_writer.write(resized)
        cv2.imshow("Color Frame", resized)
        key = cv2.waitKey(1)
        if key == 27: break

    if video_writer is not None:
        video_writer.release()
        print(f"视频已成功保存到: {output_path}")
    cv2.destroyAllWindows()
    return all_stats_data


def visualize_statistics(csv_path):
    """
    读取CSV文件并为每个统计指标生成可视化图表。
    """
    print(f"开始为 {csv_path} 生成可视化图表...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[错误] 统计文件未找到: {csv_path}")
        return

    if df.empty:
        print("[警告] 统计文件为空，跳过可视化。")
        return

    metrics_to_plot = ['area', 'aspect_ratio', 'circularity', 'pca_avg_axis_length']
    base_filename = os.path.splitext(csv_path)[0]

    for metric in metrics_to_plot:
        plt.figure(figsize=(15, 8))

        for object_id, group in df.groupby('assigned_number'):
            plt.plot(group['frame'], group[metric], marker='o', linestyle='-', markersize=4, label=f'ID {object_id}')

        plt.title(f'{metric.replace("_", " ").title()} vs. Frame Number', fontsize=16)
        plt.xlabel("Frame Number", fontsize=12)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        plt.legend(title="Object ID")
        plt.grid(True)
        plt.tight_layout()

        plot_filename = f"{base_filename}_{metric}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"图表已保存: {plot_filename}")


def main():
    selected_list = speed_1_lst

    for idx in selected_list:
        folder = datasets_path[idx]
        print(f"正在处理文件夹: {folder}")
        image_list = get_sorted_png_files(folder)
        if not image_list:
            print("[警告] 未找到PNG图片。")
            continue

        folder_name = os.path.basename(os.path.normpath(folder))
        output_video_path = f"output_{folder_name}.mp4"

        stats_data = visualize_images(image_list, save_video=SAVE_VIDEO, output_path=output_video_path)

        stats_output_path = f"stats_{folder_name}.csv"
        if stats_data:
            print(f"正在保存统计数据到: {stats_output_path}")
            fieldnames = stats_data[0].keys()
            with open(stats_output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(stats_data)
            print("统计数据保存成功。")

            visualize_statistics(stats_output_path)

        print(f"文件夹处理完毕: {folder}。")


if __name__ == "__main__":
    main()