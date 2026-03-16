import pandas as pd
import numpy as np

def main():
    # 读取原始数据
    df = pd.read_csv('baselines_result.csv')
    
    # 需要统计的指标列
    metrics = ['PR', 'RC', 'AUC', 'AP', 'F1']
    
    # 按照数据集和方法进行分组，计算均值和标准差
    grouped = df.groupby(['Dataset', 'Method'])[metrics].agg(['mean', 'std']).reset_index()
    
    # 创建一个新的DataFrame来存储"均值±标准差"的格式
    result_df = pd.DataFrame()
    result_df['Dataset'] = grouped['Dataset']
    result_df['Method'] = grouped['Method']
    
    for metric in metrics:
        # 获取均值和标准差，并保留4位小数
        mean_series = grouped[metric]['mean'].apply(lambda x: f"{x:.4f}")
        std_series = grouped[metric]['std'].apply(lambda x: f"{x:.4f}")
        
        # 拼接成 均值±标准差 的形式
        result_df[metric] = mean_series + '±' + std_series
        
        # 同时保留均值一列，标准差一列，如果需要也可以加上
        # result_df[f'{metric}_mean'] = mean_series
        # result_df[f'{metric}_std'] = std_series

    # 获取所有的数据集 (MSDS, GAIA)
    datasets = result_df['Dataset'].unique()
    
    # 尝试保存为Excel文件（需要安装 openpyxl），如果失败则保存为多个CSV
    try:
        with pd.ExcelWriter('statistical_results.xlsx', engine='openpyxl') as writer:
            for dataset in datasets:
                # 筛选对应数据集的数据
                dataset_df = result_df[result_df['Dataset'] == dataset].drop(columns=['Dataset'])
                # 保存到不同的sheet中
                dataset_df.to_excel(writer, sheet_name=dataset, index=False)
        print("成功保存为 Excel 文件: statistical_results.xlsx")
    except ImportError:
        print("未安装 openpyxl，将降级保存为多个 CSV 文件。")
        for dataset in datasets:
            dataset_df = result_df[result_df['Dataset'] == dataset].drop(columns=['Dataset'])
            csv_filename = f'{dataset}_statistical_results.csv'
            dataset_df.to_csv(csv_filename, index=False)
            print(f"成功保存为 CSV 文件: {csv_filename}")

if __name__ == '__main__':
    main()
