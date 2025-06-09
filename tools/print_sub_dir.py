"""
用于打印数据集目录下所有的子目录和文件统计数
"""
import os
from collections import defaultdict

# 数据集目录
current_dir = r"C:\Users\JmZha\VSCode_Project\Datasets\Apple"

# 遍历 {数据集目录} 下的所有子目录
for root, dirs, files in os.walk(current_dir):
    # 只处理 {当前目录} 下的 [第一层] 子目录
    if root == current_dir:
        for d in dirs:
            subdir_path = os.path.join(root, d)
            ext_count = defaultdict(int)

            # 遍历 [子目录] 中的文件
            for file in os.listdir(subdir_path):
                full_path = os.path.join(subdir_path, file)
                if os.path.isfile(full_path):
                    _, ext = os.path.splitext(file)
                    if ext:
                        ext_count[ext.lower()] += 1
                    else:
                        ext_count["<no_ext>"] += 1

            print(f"- 子目录: {d}")
            for ext, count in sorted(ext_count.items()):
                print(f"  {ext}: {count}")
