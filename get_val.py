import os
import shutil



num_files_to_move = 3  
source_directory=''
destination_directory=''
def create_matching_structure(src_dir, dst_dir):

    for root, dirs, _ in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        dst_root = os.path.join(dst_dir, rel_path)
        if not os.path.exists(dst_root):
            os.makedirs(dst_root)

def move_files_with_structure(src_dir, dst_dir, num_files=1):

    # 创建匹配的目录结构
    create_matching_structure(src_dir, dst_dir)

    # 遍历src_dir下的所有子目录
    for subdir, _, files in os.walk(src_dir):
        # 对于每个子目录中的文件
        moved_count = 0
        for file in files:
            if moved_count >= num_files:
                break
            
            
            src_path = os.path.join(subdir, file)
            
            
            rel_path = os.path.relpath(subdir, src_dir)
            dst_subdir = os.path.join(dst_dir, rel_path)
            dst_path = os.path.join(dst_subdir, file)
            
            
            shutil.move(src_path, dst_path)
            print(f"Moved {file} to {dst_path}")
            moved_count += 1



if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)



move_files_with_structure(source_directory, destination_directory, num_files_to_move)