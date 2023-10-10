import os
import shutil
import random

def copy_random_images(src_root, dst_root, n=1000):
    # 각 상위 폴더(train, val, test)를 반복
    for super_folder in ['train', 'val', 'test']:
        src_super_folder = os.path.join(src_root, super_folder)
        dst_super_folder = os.path.join(dst_root, super_folder)

        # 상위 폴더가 존재하지 않으면 생성
        os.makedirs(dst_super_folder, exist_ok=True)

        # 각 하위 폴더를 반복
        for sub_folder in os.listdir(src_super_folder):
            src_sub_folder = os.path.join(src_super_folder, sub_folder)
            dst_sub_folder = os.path.join(dst_super_folder, sub_folder)

            # 하위 폴더가 존재하지 않으면 생성
            os.makedirs(dst_sub_folder, exist_ok=True)

            # 모든 이미지 파일을 가져와서 무작위로 섞기
            all_images = [f for f in os.listdir(src_sub_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            random.shuffle(all_images)

            # 첫 1000개의 이미지를 선택하고 복사 (또는 폴더에 있는 이미지 수가 1000개 미만인 경우 모든 이미지를 복사)
            for image in all_images[:n]:
                src_image = os.path.join(src_sub_folder, image)
                dst_image = os.path.join(dst_sub_folder, image)
                shutil.copy(src_image, dst_image)

# 사용 예
src_root = '01.데이터'  # 원본 데이터셋 폴더
dst_root = 'dataset'  # 새로운 데이터셋 폴더 (데이터 구조를 복사할 위치)
copy_random_images(src_root, dst_root)
