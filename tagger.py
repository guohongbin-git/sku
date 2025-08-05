import easyocr
import csv
from pathlib import Path
import re

def get_best_guess(ocr_texts):
    """
    从OCR结果中猜测一个最可能的标签。
    一个简单的启发式规则：
    1. 优先选择包含常见品牌关键词的结果。
    2. 如果没有，选择最长的、看起来最像名称的字符串。
    3. 过滤掉纯数字、常见单位和广告语。
    """
    if not ocr_texts:
        return ""

    # 过滤掉不可能是标签的文本
    ignore_keywords = ['净含量', 'ml', '毫升', 'L', '升', '公司', '地址', '电话', '官网', '营养成分表']
    
    candidates = []
    for text in ocr_texts:
        text = text.strip()
        if any(keyword in text for keyword in ignore_keywords):
            continue
        if re.match(r'^["\d\s\.\%]+$', text): # 过滤纯数字/百分比
            continue
        candidates.append(text)

    if not candidates:
        return ocr_texts[0] # 如果都过滤掉了，返回第一个

    # 选择最长的候选者作为最佳猜测
    return max(candidates, key=len)

def main():
    """
    一个交互式的命令行工具，用于通过OCR辅助为图片打标签。
    """
    # 初始化EasyOCR，指定识别中文和英文
    # 第一次运行时会自动下载模型，请耐心等待
    print("正在加载EasyOCR模型，首次运行可能需要几分钟...")
    reader = easyocr.Reader(['ch_sim', 'en']) 
    print("模型加载完成。")

    source_dir = Path('/Users/guohongbin/projects/识别/饮料-700张')
    output_csv = Path('/Users/guohongbin/projects/识别/labels.csv')
    
    # 获取所有图片路径
    image_paths = sorted(list(source_dir.glob('**/*.jpg')))
    
    # 读取已标注的文件，避免重复工作
    processed_files = set()
    if output_csv.exists():
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader_csv = csv.reader(f)
            next(reader_csv, None) # 跳过表头
            for row in reader_csv:
                if row:
                    processed_files.add(row[0])
    
    # 如果是第一次运行，写入表头
    if not output_csv.exists():
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'label'])

    total_files = len(image_paths)
    processed_count = len(processed_files)
    
    print(f"\n共找到 {total_files} 张图片，已处理 {processed_count} 张。")
    print("开始进行标注。输入 'q' 退出，'s' 跳过当前图片。")

    try:
        for i, img_path in enumerate(image_paths):
            relative_path_str = str(img_path.relative_to(Path.cwd()))
            
            if relative_path_str in processed_files:
                continue

            print("\n" + "="*50)
            print(f"进度: [{i+1}/{total_files}]")
            print(f"正在处理图片: {img_path}")

            try:
                # OCR识别
                result = reader.readtext(str(img_path))
                ocr_texts = [item[1] for item in result]
                
                if not ocr_texts:
                    print("OCR未能识别出任何文本。")
                    user_input = input(">> 请手动输入标签 (或 's' 跳过): ").strip()
                    if user_input.lower() == 's':
                        continue
                    label = user_input
                else:
                    print("\nOCR 识别结果:")
                    for text in ocr_texts:
                        print(f"  - {text}")
                    
                    # 推荐标签
                    guess = get_best_guess(ocr_texts)
                    
                    # 获取用户确认
                    user_input = input(f"\n>> 请输入正确标签 (推荐: '{guess}', 直接回车使用推荐, 's'跳过): ").strip()
                    
                    if user_input.lower() == 'q':
                        print("用户请求退出。")
                        break
                    if user_input.lower() == 's':
                        print("跳过当前图片。")
                        continue
                    
                    label = user_input if user_input else guess

                # 将结果追加到CSV文件
                with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([relative_path_str, label])
                
                print(f"已保存标签: '{label}'")
                processed_files.add(relative_path_str)

            except Exception as e:
                print(f"处理图片 {img_path} 时发生错误: {e}")
                continue

    except KeyboardInterrupt:
        print("\n检测到中断，正在退出程序。")
    
    print("\n" + "="*50)
    print("标注流程结束。")
    print(f"总共已标注 {len(processed_files)} 张图片。")
    print(f"结果已保存至: {output_csv}")

if __name__ == '__main__':
    main()
