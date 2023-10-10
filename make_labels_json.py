import os
import json

def create_label_mapping(dataset_path, output_path):
    label_mapping = {"train": {}, "val": {}}
    
    # 'train'과 'val' 폴더를 각각 처리
    for data_type in ['train', 'val']:
        type_path = os.path.join(dataset_path, data_type)
        
        # 각 클래스 폴더에 대해
        for class_label in os.listdir(type_path):
            class_path = os.path.join(type_path, class_label)
            
            # 폴더인지 확인 (이미지가 있는 폴더만 처리)
            if os.path.isdir(class_path):
                
                # 폴더 내의 모든 이미지 파일에 대해
                for image_file in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_file)
                    
                    # 파일인지 확인
                    if os.path.isfile(image_path):
                        # 이미지 파일 이름과 라벨을 매핑
                        label_mapping[data_type][image_file] = class_label
    
    # 결과를 JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(label_mapping, outfile, ensure_ascii=False)

# 사용 예
create_label_mapping("D:/실전 ai 2/247.지능형 스마트팜(참외) 데이터/01.데이터", "y_labels.json")
