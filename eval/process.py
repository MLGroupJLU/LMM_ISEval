import os
import sys
import json
import argparse
import base64
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datasets import load_dataset
import torch

# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录的绝对路径
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
# 将父目录添加到模块搜索路径的开头
sys.path.insert(0, parent_dir)

from models.VLM import VLM

root_folder = 'data'

# 定义目标文件列表
target_files = [
    "exclusive_condition.jsonl",
    "grammatical_worded_error.jsonl",
    "irrelevant_condition.jsonl",
    "lacks_condition.jsonl",
    "misguided_logic.jsonl",
    "misuse_confusion.jsonl",
    "unclear_citation.jsonl"
]

def encode_image(image_path: str) -> str:
    """将图片编码为 base64 字符串。"""
    try:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
            logging.info(f"图片 {image_path} 已成功编码为 base64 字符串.")
            return encoded_image
    except Exception as e:
        logging.error(f"图片编码失败: {image_path}, {e}")
        return ""


def get_answer(vlm, question: str, image_path: str) -> str:
    """调用 VLM 模型获取答案，失败自动重试。"""
    base64_img = encode_image(image_path)
    if not base64_img:
        logging.error(f"未能编码图片 {image_path}, 无法继续请求模型.")
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        }
    ]

    for _ in range(5):
        try:
            logging.info(f"发送问题: {question[:60]}... 请求模型...")
            return vlm.get_response(messages)
        except Exception as e:
            logging.error(f"调用 VLM 出错: {e}, 5 秒后重试")
            time.sleep(5)
    return ""


def answer_ill_query(vlm, problem_data):
    ill_query = problem_data["ill_query"]
    return get_answer(vlm, ill_query, problem_data["image_path"])


def answer_normal_query(vlm, problem_data):
    normal_query = problem_data["normal_query"]
    return get_answer(vlm, normal_query, problem_data["image_path"])


def answer_ill_query_with_hint(vlm, problem_data):
    ill_query = problem_data["ill_query"]
    ill_query += """
Check if there are any errors in the question's premises before answering. If there are, please report them promptly. 
"""
    return get_answer(vlm, ill_query, problem_data["image_path"])


def inference(vlm, problem_data):
    inference_result_template = {
        "pid": problem_data["pid"],
        "answer_to_normal": answer_normal_query(vlm, problem_data),
        "answer_to_ill": answer_ill_query(vlm, problem_data),
        "answer_to_ill_with_hint": answer_ill_query_with_hint(vlm, problem_data)
    }
    return inference_result_template


def process_problem(vlm, cur_data):
    return inference(vlm, cur_data)


def process_jsonl_file(vlm, file_path: str, model_name: str):
    """遍历文件的每一行，按需处理并实时写入结果。"""
    base_name = os.path.basename(file_path)
    name, ext = os.path.splitext(base_name)
    # 处理模型名称，将斜杠替换为下划线
    safe_model_name = model_name.replace('/', '_')
    # 生成包含模型名字的新文件名
    new_file_name = f"{name}_{safe_model_name}{ext}"
    new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)

    with open(file_path, 'r', encoding='utf-8') as fr, open(new_file_path, 'w', encoding='utf-8') as fw:
        for line in fr:
            try:
                data = json.loads(line)
                processed_data = process_problem(vlm, data)
                # 实时写入处理后的 JSON 数据
                fw.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
                fw.flush()  # 强制刷新缓冲区，确保数据立即写入文件
            except json.JSONDecodeError:
                logging.warning(f"跳过无效 JSON 行: {line.strip()}")
                fw.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image-related questions")
    parser.add_argument("--model_name", type=str, default="your_vlm_model", help="The VLM model to use")
    parser.add_argument("--save_frequency", type=int, default=1, help="Save frequency")
    parser.add_argument("--mode", type=str, default="inference", help="Two modes: inference, check")
    parser.add_argument("--DEBUG", type=bool, default=True, help="Debug mode")
    parser.add_argument("--dataset_load_proc", type=int, default=10, help="Dataset load process")
    parser.add_argument("--infer_proc", type=int, default=3, help="Model inference process number")
    args = parser.parse_args()

    class Model_ARGS:
        temperature = 0.0
        top_p = 1
        max_tokens = 32768
        n = 1

    model_args = Model_ARGS()

    vlm = VLM(model_name=args.model_name, model_args=model_args)

    # 处理模型名称，将斜杠替换为下划线
    safe_model_name = args.model_name.replace('/', '_')
    # 动态生成日志文件名
    log_filename = f"image_process_{safe_model_name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[process] %(message)s",
        datefmt="[%X]",
        filename=log_filename
    )

    # 修改为仅处理 target_files 列表中的文件
    for root, _, files in os.walk('.'):
        for file in files:
            if file in target_files and file.endswith('.jsonl'):
                process_jsonl_file(vlm, os.path.join(root, file), args.model_name)