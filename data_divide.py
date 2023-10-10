import os
import shutil
import random

# 경로 설정
data_dir = '01.데이터'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# test 폴더 생성 (이미 존재하지 않는 경우)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# 각 클래스 폴더에 대해
for class_name in os.listdir(train_dir):
    # 클래스별 폴더 경로 설정
    class_train_dir = os.path.join(train_dir, class_name)
    class_test_dir = os.path.join(test_dir, class_name)
    
    # 클래스별 test 폴더 생성 (이미 존재하지 않는 경우)
    if not os.path.exists(class_test_dir):
        os.makedirs(class_test_dir)
    
    # 이미지 파일 리스트 가져오기
    image_files = [f for f in os.listdir(class_train_dir) if os.path.isfile(os.path.join(class_train_dir, f))]
    
    # 8:2 비율로 나누기 위해 20%의 파일을 랜덤하게 선택
    test_files = random.sample(image_files, int(len(image_files) * 0.2))
    
    # 선택된 파일을 test 폴더로 이동
    for test_file in test_files:
        shutil.move(os.path.join(class_train_dir, test_file), os.path.join(class_test_dir, test_file))
