# -*- coding: utf-8 -*-
import json
import csv
import re

input_file = r'C:\Users\HP\Desktop\xiaorong2\gemini\120questions_utf8.csv'
output_file = r'C:\Users\HP\Desktop\xiaorong2\gemini\120questions_fixed.csv'

rows = []
with open(input_file, 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    header = next(reader, None)
    print('列名:', header)
    for i, row in enumerate(reader):
        print(f'处理第{i+1}行...')
        if len(row) < 2:
            continue
        original = row[0].strip()
        sub_str = row[1].strip()
        try:
            sub_qs = json.loads(sub_str)
        except:
            matches = re.findall(r'"([^"]*)"', sub_str)
            sub_qs = [m.strip() for m in matches if m.strip()]
        if original and sub_qs:
            rows.append([original, json.dumps(sub_qs, ensure_ascii=False)])

with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['original', 'sub_questions'])
    writer.writerows(rows)

print(f'完成！共 {len(rows)} 条')