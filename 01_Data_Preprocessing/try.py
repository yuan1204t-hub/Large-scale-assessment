import os
import pandas as pd
from openpyxl import load_workbook

# --- 配置区域 ---
folder_path = r'C:\Users\仟肆\Desktop\数据部分\Data\Data (RSM_Experimental_Design_Section)'  # 替换为你的 Excel 文件夹路径
name_map = {
    "编码前": "Before",
    "编码后": "After"
}

def batch_rename_sheets(folder):
    # 遍历文件夹中所有的 excel 文件
    for filename in os.listdir(folder):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            file_path = os.path.join(folder, filename)
            
            try:
                # 使用 openpyxl 加载工作簿（为了保留格式）
                wb = load_workbook(file_path)
                changed = False
                
                for sheet in wb.worksheets:
                    if sheet.title in name_map:
                        old_name = sheet.title
                        sheet.title = name_map[old_name]
                        print(f"文件 [{filename}]: '{old_name}' -> '{sheet.title}'")
                        changed = True
                
                if changed:
                    wb.save(file_path)
                wb.close()
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 确保路径存在
    if os.path.exists(folder_path):
        batch_rename_sheets(folder_path)
        print("\n任务完成！")
    else:
        print("错误：找不到指定的文件夹路径。")