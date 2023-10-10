import json
import random

def create_random_label_mapping(input_path, output_path):
    # 가능한 라벨들
    possible_labels = ["노균병", "노균병유사", "정상", "흰가루병", "흰가루병유사"]
    
    # 기존의 라벨링 정보를 로드
    with open(input_path, 'r', encoding='utf-8') as infile:
        original_mapping = json.load(infile)
    
    # 새로운 라벨링 정보를 생성
    new_mapping = {"train": {}, "val": {}}
    
    # 'train'과 'val'에 대해 각각 처리
    for data_type in ['train', 'val']:
        for image_file, _ in original_mapping[data_type].items():
            # 랜덤으로 라벨 선택
            new_label = random.choice(possible_labels)
            new_mapping[data_type][image_file] = new_label
    
    # 새로운 라벨링 정보를 JSON 파일로 저장
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(new_mapping, outfile, ensure_ascii=False)

# 사용 예
create_random_label_mapping("D:/실전 ai 2/247.지능형 스마트팜(참외) 데이터/y_labels.json", "random_y_label.json")
