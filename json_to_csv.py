import json
import csv

def convert_json_to_csv(json_path, csv_path):
    # JSON 파일에서 라벨링 정보 로드
    with open(json_path, 'r', encoding='utf-8') as jfile:
        label_mapping = json.load(jfile)
    
    # CSV 파일로 저장
    with open(csv_path, 'w', newline='', encoding='utf-8') as cfile:
        writer = csv.writer(cfile)
        
        # 헤더 작성
        writer.writerow(["filename", "label"])
        
        # 'train'과 'val'에 대해 각각 처리
        for data_type in ['train', 'val']:
            for image_file, label in label_mapping[data_type].items():
                writer.writerow([image_file, label])

# 사용 예
convert_json_to_csv("random_y_label.json", "random_y_label.csv")
