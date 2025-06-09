from loading_datasets import datasets_path, speed_1_lst  # 可改为 speed_2_lst 或 speed_3_lst

import os
import re
import cv2
import numpy as np
# 引入scipy库来使用最高效的全局最优匹配算法（匈牙利算法）
# 如果你的环境中没有安装，请运行: pip install scipy
from scipy.optimize import linear_sum_assignment

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


def visualize_images(image_paths, window_size=(1440, 810), save_video=False, output_path="output.mp4"):
    """
    使用 OpenCV 显示图像序列，应用 HSV 模板，并结合形态学去噪和连通分量分析。
    实现跨帧连通分量颜色一致性追踪，并按要求显示累加计数器和自定义编号。
    追踪逻辑已根据“两列从下向上运行”的特性进行优化。
    新增功能: 可选择将可视化结果保存为视频。
    """
    # ... (函数内部的其他变量定义保持不变) ...
    # 定义感兴趣区域 (ROI)
    roi_x_min = 820
    roi_x_max = 1360
    roi_y_min = 100
    roi_y_max = 980

    # 定义HSV颜色阈值
    lower_hsv = np.array([10, 0, 30])
    upper_hsv = np.array([30, 255, 255])

    # 定义形态学操作的核
    morph_kernel = np.ones((5, 5), np.uint8)

    # 定义面积阈值
    min_noise_area_threshold = 2750

    # --- 追踪相关的变量 ---
    tracked_objects = {}
    next_unique_id = 0
    global_component_counter = 0
    initial_vertical_movement = 20
    max_distance_for_tracking = 100
    max_missed_frames = 5

    # --- 自定义编号相关参数 ---
    num_objects_per_column = 9
    next_left_column_number = 1
    next_right_column_number = num_objects_per_column + 1

    # 预定义颜色
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
        (64, 64, 64), (192, 192, 192)
    ]
    color_index = 0

    # --- 视频录制设置 ---
    video_writer = None
    if save_video:
        # 定义视频编码器和创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4V 编码器 for .mp4 file
        # 注意：帧大小必须与写入的帧（这里是resized）完全匹配
        video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, window_size)
        if not video_writer.isOpened():
            print(f"[错误] 无法打开视频文件进行写入: {output_path}")
            save_video = False  # 如果无法打开，则禁用保存功能
        else:
            print(f"将开始录制视频到: {output_path}")

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[ERROR] 加载失败: {path}")
            continue

        # ... (图像处理和追踪逻辑保持不变) ...
        roi_img = img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        roi_center_x_local = (roi_x_max - roi_x_min) / 2

        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        denoised_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(denoised_mask, connectivity=8)

        current_detections = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_noise_area_threshold:
                current_detections.append({
                    'centroid': centroids[i],
                    'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]),
                    'area': stats[i, cv2.CC_STAT_AREA],
                    'label_id': i
                })

        if not tracked_objects:
            for det in current_detections:
                is_right = det['centroid'][0] > roi_center_x_local
                assigned_num = next_right_column_number if is_right else next_left_column_number
                if is_right:
                    next_right_column_number += 1
                else:
                    next_left_column_number += 1
                tracked_objects[next_unique_id] = {
                    'centroid': det['centroid'], 'color': COLORS[color_index % len(COLORS)],
                    'missed_frames': 0, 'assigned_number': assigned_num,
                    'velocity': np.array([0, -initial_vertical_movement]),
                    'unique_id': next_unique_id, 'bbox': det['bbox'], 'label_id': det['label_id']
                }
                next_unique_id += 1;
                color_index += 1;
                global_component_counter += 1
        else:
            tracked_ids = list(tracked_objects.keys())
            predicted_centroids = [tracked_objects[tid]['centroid'] + tracked_objects[tid]['velocity'] for tid in
                                   tracked_ids]
            cost_matrix = np.full((len(predicted_centroids), len(current_detections)), 1e6)
            for t_idx, pred_cen in enumerate(predicted_centroids):
                for d_idx, det in enumerate(current_detections):
                    dist = np.linalg.norm(pred_cen - det['centroid'])
                    if dist < max_distance_for_tracking: cost_matrix[t_idx, d_idx] = dist
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            new_tracked_objects = {}
            matched_detection_indices = set()
            for t_idx, d_idx in zip(row_ind, col_ind):
                if cost_matrix[t_idx, d_idx] < max_distance_for_tracking:
                    tid = tracked_ids[t_idx];
                    detection = current_detections[d_idx];
                    old_data = tracked_objects[tid]
                    new_velocity = detection['centroid'] - old_data['centroid']
                    updated_velocity = old_data['velocity'] * 0.5 + new_velocity * 0.5
                    new_tracked_objects[tid] = {
                        'centroid': detection['centroid'], 'color': old_data['color'], 'missed_frames': 0,
                        'assigned_number': old_data['assigned_number'], 'velocity': updated_velocity,
                        'unique_id': tid, 'bbox': detection['bbox'], 'label_id': detection['label_id']
                    }
                    matched_detection_indices.add(d_idx)
            unmatched_track_ids = set(tracked_ids) - {tracked_ids[t_idx] for t_idx, d_idx in zip(row_ind, col_ind) if
                                                      cost_matrix[t_idx, d_idx] < max_distance_for_tracking}
            for tid in unmatched_track_ids:
                obj = tracked_objects[tid];
                obj['missed_frames'] += 1
                if obj['missed_frames'] <= max_missed_frames:
                    obj['centroid'] += obj['velocity'];
                    new_tracked_objects[tid] = obj
            unmatched_detection_indices = set(range(len(current_detections))) - matched_detection_indices
            for d_idx in unmatched_detection_indices:
                det = current_detections[d_idx];
                is_right = det['centroid'][0] > roi_center_x_local
                assigned_num = next_right_column_number if is_right else next_left_column_number
                if is_right:
                    next_right_column_number += 1
                else:
                    next_left_column_number += 1
                new_tracked_objects[next_unique_id] = {
                    'centroid': det['centroid'], 'color': COLORS[color_index % len(COLORS)], 'missed_frames': 0,
                    'assigned_number': assigned_num, 'velocity': np.array([0, -initial_vertical_movement]),
                    'unique_id': next_unique_id, 'bbox': det['bbox'], 'label_id': det['label_id']
                }
                next_unique_id += 1;
                color_index += 1;
                global_component_counter += 1
            tracked_objects = new_tracked_objects

        # --- 可视化 ---
        processed_img = np.full(img.shape, 255, dtype=np.uint8)
        colored_output_roi = np.zeros_like(roi_img)
        for obj in tracked_objects.values():
            if obj['missed_frames'] == 0:
                colored_output_roi[labels == obj['label_id']] = obj['color']
                (x, y, w, h) = obj['bbox'];
                text_to_display = str(obj['assigned_number'])
                text_x = int(x + w + 5);
                text_y = int(y + h / 2)
                if text_x + 50 > colored_output_roi.shape[1] and x - 50 > 0: text_x = int(x - 50)
                cv2.putText(colored_output_roi, text_to_display, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        processed_img[roi_y_min:roi_y_max, roi_x_min:roi_x_max] = colored_output_roi
        text_total_components = f"Detected in Total: {global_component_counter}"
        cv2.putText(processed_img, text_total_components, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        resized = cv2.resize(processed_img, window_size)

        # --- 写入视频帧 ---
        if save_video and video_writer is not None:
            video_writer.write(resized)

        cv2.imshow("Color Frame", resized)
        key = cv2.waitKey(1)
        if key == 27:
            break

    # --- 循环结束后，释放资源 ---
    if video_writer is not None:
        video_writer.release()
        print(f"视频已成功保存到: {output_path}")

    cv2.destroyAllWindows()


def main():
    # 确保你已经安装了scipy: pip install scipy
    selected_list = speed_1_lst

    for idx in selected_list:
        folder = datasets_path[idx]
        print(f"正在处理文件夹: {folder}")
        image_list = get_sorted_png_files(folder)
        if not image_list:
            print("[警告] 未找到PNG图片。")
            continue

        # 为视频文件创建一个基于文件夹名的唯一名称
        folder_name = os.path.basename(os.path.normpath(folder))
        output_video_path = f"output_{folder_name}.mp4"

        # 将视频录制开关和路径传递给处理函数
        visualize_images(image_list, save_video=SAVE_VIDEO, output_path=output_video_path)

        print(f"文件夹处理完毕: {folder}。")


if __name__ == "__main__":
    main()