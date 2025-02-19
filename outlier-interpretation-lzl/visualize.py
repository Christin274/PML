import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import numpy as np
import pandas as pd
from config import root
import ast
import itertools


def main(path):
    data_name = path.split("/")[-1].split(".")[0] #get data name
    data_name = data_name[3:]
    # print(data_name)
    df = pd.read_csv(path) #get orginal data with whole features
    total_columns = df.shape[1]
    extracted_data = pd.DataFrame() #create a new space to store the extracted data with certain featueres
    available_indices = list(set(range(total_columns)))
    all_combinations = list(itertools.combinations(available_indices, 3))
    for i in range(1):
        new_indices = all_combinations[i]
                # print(indices,type(indices))
                # print(type(index) for index in indices)
        if True:
            # indices = [int(idx) for idx in new_indices]
            indices = [13,19,25]
            extracted_data = df.iloc[:,indices]
            extracted_data = extracted_data.copy()
            if len(extracted_data.columns)==3:
                for i in range(len(extracted_data.columns)):
                    extracted_data[f'Add_{i+1+len(extracted_data.columns)}'] = 0
            label = df.iloc[:,-1] #get the label of the anomaly
            # print(label)
            extracted_data['label'] = label.values

            extracted_data_0 = extracted_data[extracted_data['label'] == 0].sample(n=20, random_state=1)
            extracted_data_1 = extracted_data[extracted_data['label'] == 1].sample(n=30, random_state=1)
            final_extracted_data = pd.concat([extracted_data_0, extracted_data_1])

            print(final_extracted_data)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
                    
                    # 设置颜色
            colors = []
            for index, row in final_extracted_data.iterrows():
                if row['label'] == 1:  # 标签为1的样本
                    colors.append('black')
                else:  # 标签为0的样本
                    colors.append('green')

            for col in range(3):  # 假设你的 DataFrame 名为 final_extracted_data
                final_extracted_data.iloc[:, col] = pd.to_numeric(final_extracted_data.iloc[:, col], errors='coerce')

            ax.scatter(final_extracted_data.iloc[:, 0], final_extracted_data.iloc[:, 1], final_extracted_data.iloc[:, 2], c=colors)
            ax.set_xlabel(indices[0])
            ax.set_ylabel(indices[1])
            ax.set_zlabel(indices[2])
            ax.set_title('3D Data Visualization')
            plt.savefig(f"pic_3d_result/{data_name}_{indices[0]}_{indices[1]}_{indices[2]}.png")

            combinations = list(itertools.combinations(range(3), 2))  # 获取所有两两组合
                    
            for idx_combination in combinations:
                col1 = final_extracted_data.iloc[:, idx_combination[0]]
                col2 = final_extracted_data.iloc[:, idx_combination[1]]

                fig2, ax2 = plt.subplots()
                ax2.scatter(col1, col2, c=colors)
                ax2.set_xlabel(f'Feature {indices[idx_combination[0]]}')
                ax2.set_ylabel(f'Feature {indices[idx_combination[1]]}')
                ax2.set_title('2D Data Visualization')

                        # 为标签为1的点添加编号
                for i, row in final_extracted_data.iterrows():
                    if row['label'] == 1 or row["label"]==0:
                        ax2.annotate(str(i), (row[idx_combination[0]], row[idx_combination[1]]), textcoords="offset points", xytext=(0, 10), ha='center')

                        # 保存图片，使用组合的指标命名
                plt.savefig(f"pic_result/{data_name}_{indices[idx_combination[0]]}_{indices[idx_combination[1]]}.png")

    
    return

if __name__ == '__main__':
    input_root_list = [root + "test_data/"] #to get data
    runs = 1

    for input_root in input_root_list:
        if os.path.isdir(input_root):
            for file_name in sorted(os.listdir(input_root)):
                if file_name.endswith(".csv"):
                    input_path = str(os.path.join(input_root, file_name))
                    name = input_path.split("/")[-1].split('.')[0]
                    main(input_path)

        else:
            input_path = input_root
            name = input_path.split("/")[-1].split(".")[0]
            main(input_path)
