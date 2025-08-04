import os
import json
path = "/pfs/pfs-r36Cge/qxg/datasets/procthor/0703-test/train/000"
# "/pfs/pfs-r36Cge/qxg/datasets/procthor/0703-test/train/000/"
# "/pfs/pfs-r36Cge/qxg/datasets/procthor/0702-trajectory-train/train/004"
# "/pfs/pfs-r36Cge/qxg/datasets/procthor/0701-trajectory100/train/"

success_object_count = 0
frame_count = 0

success_object_count_0 = 0
frame_count_0 = 0
success_object_count_1 = 0
frame_count_1 = 0
success_object_count_2 = 0
frame_count_2 = 0
success_object_count_3 = 0
frame_count_3 = 0
success_object_count_4 = 0
frame_count_4 = 0
success_object_count_5 = 0
frame_count_5 = 0
success_object_count_L = 0
frame_count_L = 0


entries = os.listdir(path)
houses = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]


for root, dirs, files in os.walk(path):
        # os.walk 会递归遍历所有子目录，但我们只需要检查一级子目录
        break  # 只执行一次循环，获取一级子目录
 
for dir_name in dirs:
    first_level_path = os.path.join(root, dir_name)
    # print(first_level_path)
    
    # 确保路径是一个目录
    if os.path.isdir(first_level_path):
        # 获取一级子目录中的二级子目录
        second_level_dirs = [d for d in os.listdir(first_level_path) if os.path.isdir(os.path.join(first_level_path, d))]
        # 统计二级子文件夹数量
        success_object_count += len(second_level_dirs)

        for second_level_dir in second_level_dirs:
            mode = second_level_dir.split('|')[2]
            
            
            second_level_path = os.path.join(first_level_path, second_level_dir)
            metadata_path = os.path.join(second_level_path, "metadata")
            actions_file_path = os.path.join(metadata_path, "actions.json")

            # 检查 metadata 文件夹和 actions.json 文件是否存在
            if os.path.isdir(metadata_path) and os.path.isfile(actions_file_path):
                try:
                    with open(actions_file_path, 'r') as f:
                        actions = json.load(f)
                        if isinstance(actions, list):
                            frame_count += len(actions)
                            if mode == '1':
                                success_object_count_1 +=1
                                frame_count_1 +=len(actions)
                            elif mode == '2':
                                success_object_count_2 +=1
                                frame_count_2 +=len(actions)
                            elif mode == '3':
                                success_object_count_3 +=1
                                frame_count_3 +=len(actions)
                            elif mode == '4':
                                success_object_count_4 +=1
                                frame_count_4 +=len(actions)
                            elif mode == '0':
                                success_object_count_0 +=1
                                frame_count_0 +=len(actions)
                            elif mode == '5':
                                success_object_count_5 +=1
                                frame_count_5 +=len(actions)
                            else:
                                # print(second_level_path)
                                success_object_count_L +=1
                                frame_count_L +=len(actions)
                        else:
                            print(f"文件 {actions_file_path} 中的内容不是一个列表")
                except json.JSONDecodeError:
                    print(f"文件 {actions_file_path} 无法解析为 JSON")
                except Exception as e:
                    print(f"读取文件 {actions_file_path} 时发生错误: {e}")

print("############ All ###############")
print("house_count:", len(houses))
print("success_object_count:",success_object_count)
print("frame_count:", frame_count)

print("############ count by tag ###############")
print("tag 0:")
print(f"success_object_count:{success_object_count_0}, frame_count:{frame_count_0}")
print("tag 1:")
print(f"success_object_count:{success_object_count_1}, frame_count:{frame_count_1}")
print("tag 2:")
print(f"success_object_count:{success_object_count_2}, frame_count:{frame_count_2}")
print("tag 3:")
print(f"success_object_count:{success_object_count_3}, frame_count:{frame_count_3}")
print("tag 4:")
print(f"success_object_count:{success_object_count_4}, frame_count:{frame_count_4}")
print("tag 5:")
print(f"success_object_count:{success_object_count_5}, frame_count:{frame_count_5}")
print("tag other:")
print(f"success_object_count:{success_object_count_L}, frame_count:{frame_count_L}")




# path = "/home/libo/program/l3cprocthor/libo/datasets_test_4/train/001"

# success_object_count = 0
# frame_count = 0

# for root, dirs, files in os.walk(path):
#         # os.walk 会递归遍历所有子目录，但我们只需要检查一级子目录
#         break  # 只执行一次循环，获取一级子目录
 
# for dir_name in dirs:
#     first_level_path = os.path.join(root, dir_name)
    
#     # 确保路径是一个目录
#     if os.path.isdir(first_level_path):
#         # 获取一级子目录中的二级子目录
#         second_level_dirs = [d for d in os.listdir(first_level_path) if os.path.isdir(os.path.join(first_level_path, d))]
#         # 统计二级子文件夹数量
#         success_object_count += len(second_level_dirs)

#         for second_level_dir in second_level_dirs:
#             second_level_path = os.path.join(first_level_path, second_level_dir)
#             metadata_path = os.path.join(second_level_path, "metadata")
#             actions_file_path = os.path.join(metadata_path, "actions.json")

#             # 检查 metadata 文件夹和 actions.json 文件是否存在
#             if os.path.isdir(metadata_path) and os.path.isfile(actions_file_path):
#                 try:
#                     with open(actions_file_path, 'r') as f:
#                         actions = json.load(f)
#                         if isinstance(actions, list):
#                             frame_count += len(actions)
#                         else:
#                             print(f"文件 {actions_file_path} 中的内容不是一个列表")
#                 except json.JSONDecodeError:
#                     print(f"文件 {actions_file_path} 无法解析为 JSON")
#                 except Exception as e:
#                     print(f"读取文件 {actions_file_path} 时发生错误: {e}")

# print("############  3 ###############")
# print("success_object_count:",success_object_count)
# print("frame_count:", frame_count)