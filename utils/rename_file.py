import os

def rename_files(directory, template="file"):
    """
    Rename all image files in the specified directory to sequential numbers (e.g., 1.jpg, 2.jpg).
    
    Args:
        directory (str): Path to the directory containing images to rename.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # 디렉터리 내 파일을 가져옴
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # 순차적으로 파일 이름 변경
    for idx, file_name in enumerate(sorted(files), start=1):
        old_path = os.path.join(directory, file_name)
        # 기존 파일의 확장자 유지
        ext = os.path.splitext(file_name)[1]
        new_name = f"{template}_{idx}{ext}"
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")
    
    print(f"Successfully renamed {len(files)} files in '{directory}'.")

    
if __name__ == "__main__":
    classes = ['aibao','fubao','huibao','lebao','ruibao']

    sorce_dir = "all_image"
    for class_name in classes:
        rename_files(os.path.join(sorce_dir,class_name), template=class_name)