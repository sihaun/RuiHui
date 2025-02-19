'''
all_image 디렉터리에 있는 이미지를 image_data 디렉터리를 만든 후 train, val 로 분배
1. 분배할 class를 받음. class = ['hui', 'rui']
2. image_data 디렉터리 생성
3. image_data/train, image_data/val 디렉터리 생성
4. 각 디렉터리 안에 len(class) 만큼 디렉터리 생성 후 그 class의 이름대로 디렉터리 생성.
5. train:val 만큼 분배한 이미지대로 class이름이 적힌 디렉터리에 분배.
len(class) 만큼 디렉터리 생성 후 
'''
import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_directories(output_dir, classes):
    """
    Create the required directory structure for the output.

    Args:
        output_dir (str): Base directory for the output.
        classes (list): List of class names.
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    # 이미 image_data 디렉터리가 존재하면 삭제
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    else:
        os.makedirs(output_dir, exist_ok=True)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    return train_dir, val_dir

def copy_files(file_list, source, destination, rename=False):
    """
    Copy files from the source directory to the destination directory.

    Args:
        file_list (list): List of filenames to copy.
        source (str): Source directory.
        destination (str): Destination directory.
    """
    if rename == False:
        for file_name in file_list:
            src_path = os.path.join(source, file_name)
            dest_path = os.path.join(destination, file_name)
            shutil.copy2(src_path, dest_path)

    else:
        # 순차적으로 파일 이름 변경
        for idx, file_name in enumerate(sorted(file_list), start=1):
            src_path = os.path.join(source, file_name)
            dest_path = os.path.join(destination, file_name)
            shutil.copy2(src_path, dest_path)
        
            old_path = os.path.join(destination, file_name)
            # 기존 파일의 확장자 유지
            ext = os.path.splitext(file_name)[1]
            new_name = f"{idx}{ext}"
            new_path = os.path.join(destination, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {file_name} -> {new_name}")

        print(f"Successfully renamed {len(file_list)} files in '{destination}'.")

def distribute_images(source_dir, output_dir, classes, train_ratio=0.8, rename=False):
    """
    Distribute images from the source directory into train and validation sets.

    Args:
        source_dir (str): Directory containing class subdirectories with images.
        output_dir (str): Directory to store train and validation sets.
        classes (list): List of class names to process.
        train_ratio (float): Ratio of images to include in the train set (default is 0.8).
    """
    # Create the directory structure
    train_dir, val_dir = prepare_directories(output_dir, classes)

    # Process each class
    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: {class_name} 클래스의 소스 디렉터리가 존재하지 않습니다. 건너뜁니다.")
            continue
        
        # Get list of images for the class
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Split images into train and validation sets
        train_images, val_images = train_test_split(images, test_size=1-train_ratio, random_state=123)
        
        # Copy files to respective directories
        copy_files(train_images, class_dir, os.path.join(train_dir, class_name), rename=rename)
        copy_files(val_images, class_dir, os.path.join(val_dir, class_name), rename=rename)

        print(f"{class_name} 클래스: Train {len(train_images)}개, Val {len(val_images)}개 복사 완료.")
    
    print("이미지 분배 완료.")

if __name__ == "__main__":
        # 분류하려는 클래스의 리스트 작성
    classes = ['can','glass','paper','plastic']


    src_dir = 'Recycle_Classification_Dataset'
    data_dir = 'recycle_dataset'

    distribute_images(source_dir=src_dir, output_dir=data_dir, classes=classes, train_ratio=0.8, rename=True)