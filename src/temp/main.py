import os
import json


def make_hashable(obj):
    """
    递归地将一个潜在的不可哈希对象（如包含列表的字典）转换成可哈希的格式。
    - 字典会被转换为键值对排序后的元组。
    - 列表会被转换为元组。
    """
    if isinstance(obj, dict):
        # 对字典，先递归处理它的值，然后将键值对排序并转换为元组
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        # 对列表，递归处理它的每个元素，然后将整个列表转换为元组
        return tuple(make_hashable(e) for e in obj)
    # 对于已经是可哈希的类型（如字符串、数字、元组等），直接返回
    return obj


def deduplicate_json_in_directory(input_path, output_file_path):
    """
    遍历指定目录下的所有JSON文件，对文件中的JSON对象列表进行去重，
    并将结果写入一个新的JSON文件。
    此版本修复了当JSON对象中包含列表时导致的 "unhashable type: 'list'" 错误。
    """
    # 使用一个集合（set）来存储已经见过的对象的唯一标识，以高效地检查重复
    seen_objects_representation = set()

    # 使用一个列表来存储所有去重后的JSON对象
    unique_objects = []

    print(f"开始扫描目录: {input_path}")

    # 1. 遍历指定路径下的所有文件和目录
    for filename in os.listdir(input_path):
        # 2. 检查文件后缀是否为.json
        if filename.endswith(".json"):
            file_path = os.path.join(input_path, filename)
            print(f"正在处理文件: {file_path}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 3. 读取并解析JSON文件内容
                    data = json.load(f)

                    if not isinstance(data, list):
                        print(f"  -> 警告: 文件 {filename} 的内容不是一个列表，已跳过。")
                        continue

                    # 4. 遍历列表中的每一个JSON对象
                    for obj in data:
                        # 使用新的辅助函数将整个对象（包括其嵌套内容）转换为可哈希的表示形式
                        obj_representation = make_hashable(obj)

                        # 5. 如果该对象的表示形式不在我们的集合中，说明是新对象
                        if obj_representation not in seen_objects_representation:
                            # 将其表示形式添加到集合中
                            seen_objects_representation.add(obj_representation)
                            # 将原始的JSON对象添加到最终结果列表中
                            unique_objects.append(obj)

            except json.JSONDecodeError:
                print(f"  -> 错误: 文件 {filename} 不是一个有效的JSON文件，已跳过。")
            except Exception as e:
                print(f"  -> 处理文件 {filename} 时发生未知错误: {e}")

    # 6. 将所有去重后的对象写入到指定的输出文件中
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # 使用indent=4格式化输出，使其更易读
            json.dump(unique_objects, f, ensure_ascii=False, indent=4)

        print("\n处理完成！")
        print(f"共找到 {len(unique_objects)} 个独立的对象。")
        print(f"结果已保存至: {output_file_path}")

    except Exception as e:
        print(f"\n写入输出文件时发生错误: {e}")


# --- 使用说明 ---
if __name__ == "__main__":
    # 1. 设置你的JSON文件所在的文件夹路径
    # 例如: "C:/Users/YourUser/Desktop/json_data"
    # 或者在Linux/macOS上: "/home/user/json_data"
    json_directory = "../../data/bad_woman"

    # 2. 设置你想要保存结果的文件名和路径
    output_file = "../../data/all_samples.json"

    # 检查路径是否存在
    if not os.path.isdir(json_directory):
        print(f"错误: 目录 '{json_directory}' 不存在。请检查路径是否正确。")
    else:
        # 调用函数开始执行
        deduplicate_json_in_directory(json_directory, output_file)
