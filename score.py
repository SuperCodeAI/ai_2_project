import csv

# CSV 파일들의 경로
a_file_path = "random_y_label.csv"     # 예측 값
b_file_path = "y_labels.csv"       # 실제 값

# CSV 파일을 읽어서 {이미지: 레이블} 형태의 딕셔너리로 변환하는 함수
def csv_to_dict(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 헤더 건너뛰기
            return {rows[0]: rows[1] for rows in reader}
    except Exception as e:
        print(f"Error reading {filepath}: {str(e)}")
        return {}

# 두 딕셔너리를 비교하여 일치하는 레이블의 개수를 반환하는 함수
def compare_dicts(dict_a, dict_b):
    matched_count = 0
    for key, value in dict_a.items():
        if key in dict_b and dict_b[key] == value:
            matched_count += 1
    return matched_count

a_dict = csv_to_dict(a_file_path)
b_dict = csv_to_dict(b_file_path)

# 두 딕셔너리가 비어있지 않은 경우에만 비교를 수행
if a_dict and b_dict:
    correct_answers = compare_dicts(a_dict, b_dict)

    # 결과 계산
    score = float(correct_answers) / len(a_dict)

    # 결과 출력
    print(f"Total Images in A: {len(a_dict)}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Score: {score:.5f}/1.00000")
else:
    print("Comparison failed due to previous errors.")
