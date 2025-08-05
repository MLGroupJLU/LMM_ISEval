import json
import os
import random
import re
import argparse
from prompt_template import prompt_templates
from models.VLM import LLM  # 使用封装好的 QwenVL 多模态模型
import logging
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 根文件夹路径
root_folder = 'data_construction'


def synthesis_final_question(llm, problem_data):
    # 获取图片路径
    image_path = problem_data["image_path"]
    # 原始问题
    question = problem_data["meta_info"]["original_question"].replace('{', '{{').replace('}', '}}')
    # 获取 prompt 模板
    conflict_type = problem_data['conflict_type']
    prompt_template = prompt_templates[conflict_type]
    # 处理
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": prompt_template.format(question=question, image_path=image_path)
        }
    ]
    tmp_cnt = 0
    while tmp_cnt < 10:
        try:
            response = llm.get_response(messages)
            pattern = r'```json\s*([\s\S]*?)\s*```'
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_data = match.group(1)
                json_data = json.loads(json_data)
                problem_data["ill_query"] = json_data['recomposed_question']
                problem_data["normal_query"] = problem_data["meta_info"]["original_question"]
                problem_data["conflict"]['original_premise'] = json_data['original_premise']
                problem_data["conflict"]['recomposed_premise'] = json_data['contradictory_premise']
                problem_data["conflict"]['conflict_reason'] = json_data['conflict_reason']
                break
        except Exception as e:
            problem_data["ill_query"] = ""
            logging.info(f"pid:{problem_data['pid']} synthesis failed!")
            tmp_cnt += 1
    return problem_data


def write_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_jsonl_file(file_path, expected_conflict_type, llm):
    """
    处理单个 jsonl 文件，补全其中的数据
    :param file_path: jsonl 文件路径
    :param expected_conflict_type: 预期的 conflict_type
    :param llm: 模型实例
    """
    updated_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line)
                conflict_type = data['conflict_type']
                # 验证 conflict_type 是否和文件夹名一致
                if conflict_type != expected_conflict_type:
                    logging.warning(f"{file_path} 中的 conflict_type {conflict_type} 与文件夹名 {expected_conflict_type} 不一致，跳过该条数据")
                    updated_lines.append(line)
                    continue

                if conflict_type not in prompt_templates:
                    logging.warning(f"未找到 {conflict_type} 对应的 prompt 模板，跳过该条数据")
                    updated_lines.append(line)
                    continue

                data = synthesis_final_question(llm, data)
                updated_lines.append(json.dumps(data, ensure_ascii=False) + '\n')

        # 将更新后的数据写回文件
        with open(file_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(updated_lines)
        logging.info(f"已完成对 {file_path} 的处理")
    except Exception as e:
        logging.error(f"处理文件 {file_path} 时出错: {e}")


def run(local_rank, args):
    llm = LLM(model_name="Qwen/Qwen2.5-VL-3B-Instruct", local_rank=local_rank)

    # 检查根文件夹是否存在
    if not os.path.exists(root_folder):
        logging.error(f"{root_folder} 文件夹不存在，请检查路径。")
        return

    # 遍历 data_construction 下的所有子文件夹
    files = []
    for large_type_folder in os.listdir(root_folder):
        large_type_folder_path = os.path.join(root_folder, large_type_folder)
        if not os.path.isdir(large_type_folder_path):
            continue

        for theme_folder in os.listdir(large_type_folder_path):
            theme_folder_path = os.path.join(large_type_folder_path, theme_folder)
            if not os.path.isdir(theme_folder_path):
                continue

            for file in os.listdir(theme_folder_path):
                if file.endswith('.jsonl'):
                    file_path = os.path.join(theme_folder_path, file)
                    files.append((file_path, theme_folder))

    # 使用 DistributedSampler 分配任务
    sampler = DistributedSampler(files, num_replicas=args.world_size, rank=local_rank)
    sampler.set_epoch(0)
    dataloader = DataLoader(files, sampler=sampler, batch_size=1)

    for file_path, expected_conflict_type in dataloader:
        process_jsonl_file(file_path[0], expected_conflict_type[0], llm)


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", type=str, default="synthesis",
                        help="two modes: synthesis, check")  # synthesis: 生成前提，check: 重新从没有被正确提取的题目中提取前提
    parser.add_argument("--save_frequency", type=int, default=2, help="")
    parser.add_argument("--DEBUG", type=bool, default=False, help="")
    parser.add_argument("--world_size", type=int, default=4, help="Number of GPUs")
    args = parser.parse_args()

    mp.spawn(run, args=(args,), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()