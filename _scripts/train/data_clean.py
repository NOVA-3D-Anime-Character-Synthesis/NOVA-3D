import os
import csv

from _util.util_v1 import * ; import _util.util_v1 as uutil


ap = uutil.argparse.ArgumentParser()
ap.add_argument('--dataset')
ap.add_argument('--phase')
args = ap.parse_args()
dataset = args.dataset
inferquery = args.phase
assert dataset == "rutileEB" or "human_rutileEB"
assert inferquery == "train" or inferquery == "test" or inferquery == "all" or inferquery == "val"
# 指定 CSV 文件路径
csv_file_read_path = f"./_data/lustrous/subsets/{dataset}_{inferquery}_total.csv"
csv_file_write_path = f"./_data/lustrous/subsets/{dataset}_{inferquery}.csv"
def read_csv_file(file_read_path, file_write_path, dataset_name):
    data = []
    count = 0
    with open(file_read_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # 查看当前数据是否存在，如果存在就写入到，如果不存在，就
            data_path = f"./_data/lustrous/renders/{dataset_name}/ortho/{row[0][-1]}/{row[0]}"
            if os.path.exists(data_path):
                data.append(row[0])
            else:
                count+=1
                # print(row)  # 处理每一行数据
    print(data)
    print(count)
    with open(file_write_path,'w',newline='') as file:
        writer = csv.writer(file,delimiter='\n')
        writer.writerow(data)
# 调用函数读取 CSV 文件
read_csv_file(csv_file_read_path,csv_file_write_path, dataset[:-1])

