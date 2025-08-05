import os
import json
import logging
import argparse
import asyncio
from typing import Dict, Any
from models.VLM import VLM

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMConflictEvaluator:
    def __init__(self, eval_model: VLM):
        self.eval_model = eval_model

    def get_evaluation_prompt(self, metric_type: str, **kwargs) -> str:
        """
        根据不同的评估指标生成相应的提示词
        """
        if metric_type == "active_conflict_detection":
            return f"""
请评估模型在没有任何提示的情况下，是否能够主动识别问题中的前提冲突。

原始问题的矛盾前提：{kwargs.get('conflict_info', '')}

模型的回答：
{kwargs.get('answer_to_ill', '')}

请根据以下标准进行评估：
1. 模型是否主动识别出了前提中的冲突或不一致？
2. 模型是否明确指出了问题的逻辑错误？
3. 模型是否回避了冲突，直接基于某一模态信息给出答案？

请按照以下格式输出JSON结果：
{{
    "detected": true/false
}}
"""

        elif metric_type == "conflict_hint_response":
            return f"""
请评估模型在明确提示存在前提冲突后的响应能力。

原始问题的矛盾前提：{kwargs.get('conflict_info', '')}

模型在提示后的回答：
{kwargs.get('answer_to_ill_with_hint', '')}

请根据以下标准进行评估：
1. 模型是否正确识别并明确指出了前提中的冲突？
2. 模型是否提供了具体的错误分析？
3. 模型是否给出了修正建议或解释？
4. 模型的响应是否全面且准确？

请按照以下格式输出JSON结果：
{{
    "responded": true/false
}}
"""

        elif metric_type == "modality_preference":
            return f"""
请评估模型在图文前提矛盾时对不同模态信息的依赖偏好。

原始问题的矛盾前提：{kwargs.get('conflict_info', '')}

模型的回答：
{kwargs.get('answer_to_ill', '')}

请根据以下标准进行评估：
1. 模型更倾向于相信图像信息还是文本信息？
2. 模型的答案主要基于哪个模态的信息？
3. 模型是否同时考虑了两个模态的信息？

请按照以下格式输出JSON结果：
{{
    "image_preference": 0/1,
    "text_preference": 0/1
}}

注意：image_preference 和 text_preference 只能有一个为 1，若无法判断两者倾向则都为 0。
"""

        elif metric_type == "cross_modal_sensitivity":
            return f"""
请评估模型对跨模态冲突的敏感度和信息融合能力。

原始问题的矛盾前提：{kwargs.get('conflict_info', '')}

模型无提示时的回答：
{kwargs.get('answer_to_ill', '')}

模型有提示时的回答：
{kwargs.get('answer_to_ill_with_hint', '')}

请根据以下标准进行评估：
1. 模型是否能明确指出图像与文本之间的语义冲突？
2. 模型是否尝试整合或解释两个模态的信息？
3. 模型是否提供了跨模态冲突的具体分析？
4. 模型是否回避了冲突，而不是正面处理？

请按照以下格式输出JSON结果：
{{
    "sensitive": true/false
}}
"""

    async def evaluate_single_metric(self, metric_type: str, **kwargs) -> Dict[str, Any]:
        """
        评估单个指标
        """
        prompt = self.get_evaluation_prompt(metric_type, **kwargs)

        messages = [
            {"role": "system", "content": "你是一个专业的AI评估专家，擅长分析模型的跨模态冲突识别能力。请严格按照要求格式输出JSON结果。"},
            {"role": "user", "content": prompt}
        ]

        try:
            result_text = self.eval_model.get_response(messages)
            # 提取JSON部分
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group())
                return result_json
            else:
                return {"error": "无法解析评估结果", "raw_response": result_text}

        except Exception as e:
            return {"error": str(e)}

    async def evaluate_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个样本的所有指标
        """
        # 准备冲突信息
        conflict_info = ""
        if 'conflict' in item:
            conflict_info = f"原始前提：{item['conflict'].get('original_premise', '')}\n"
            conflict_info += f"矛盾前提：{item['conflict'].get('recomposed_premise', '')}\n"
            conflict_info += f"冲突原因：{item['conflict'].get('conflict_reason', '')}"

        # 准备评估参数
        eval_params = {
            'conflict_info': conflict_info,
            'answer_to_ill': item.get('answer_to_ill', ''),
            'answer_to_ill_with_hint': item.get('answer_to_ill_with_hint', ''),
            'answer_to_normal': item.get('answer_to_normal', '')
        }

        # 并行评估所有指标
        tasks = [
            self.evaluate_single_metric("active_conflict_detection", **eval_params),
            self.evaluate_single_metric("conflict_hint_response", **eval_params),
            self.evaluate_single_metric("modality_preference", **eval_params),
            self.evaluate_single_metric("cross_modal_sensitivity", **eval_params)
        ]

        results = await asyncio.gather(*tasks)

        return {
            'pid': item.get('pid'),
            'active_conflict_detection': results[0],
            'conflict_hint_response': results[1],
            'modality_preference': results[2],
            'cross_modal_sensitivity': results[3]
        }


def load_initial_data(initial_file_path):
    """
    加载单个 initial JSONL 文件中的原始数据
    :param initial_file_path: initial JSONL 文件路径
    :return: 包含原始数据的字典，键为 pid
    """
    data_dict = {}
    with open(initial_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            data_dict[data["pid"]] = data
    return data_dict


def load_model_answers(model_folder, initial_file_name, model_name):
    """
    加载指定模型文件夹中与 initial 文件对应的回答数据
    :param model_folder: 指定模型文件夹路径
    :param initial_file_name: initial 文件夹中的文件名
    :param model_name: 模型名称
    :return: 包含回答数据的字典，键为 pid
    """
    base_name, _ = os.path.splitext(initial_file_name)
    model_file_name = f"{base_name}_{model_name}.jsonl"
    model_file_path = os.path.join(model_folder, model_file_name)

    answer_dict = {}
    if os.path.exists(model_file_path):
        with open(model_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                answer_dict[data["pid"]] = data
    else:
        logging.warning(f"未找到对应的 {model_name} 文件: {model_file_path}")
    return answer_dict


async def evaluate_file(initial_file_path, model_folder, model_name, eval_model_name, output_file):
    """
    评估单个 JSONL 文件的指标
    :param initial_file_path: initial JSONL 文件路径
    :param model_folder: 指定模型文件夹路径
    :param model_name: 被评估的模型名称
    :param eval_model_name: 用于评估的模型名称
    :param output_file: 输出结果文件
    :return: 指标的评估结果
    """
    try:
        # 初始化评估模型客户端，设置评估模型为 o3
        eval_model_name = "o3"
        llm = VLM(eval_model_name, {})
        evaluator = LLMConflictEvaluator(llm)

        initial_data = load_initial_data(initial_file_path)
        model_answers = load_model_answers(model_folder, os.path.basename(initial_file_path), model_name)

        total_active_recognition_count = 0
        total_conflict_hint_response_count = 0
        total_image_preference_count = 0
        total_text_preference_count = 0
        total_cross_modal_sensitivity_count = 0
        total_samples = 0

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"文件: {os.path.basename(initial_file_path)}\n")

            for pid in initial_data:
                if pid not in model_answers:
                    logging.warning(f"PID {pid} 没有对应的回答数据，跳过")
                    continue

            item = {**initial_data[pid], **model_answers[pid]}
            evaluation_result = await evaluator.evaluate_single_item(item)

            # 主动冲突识别率
            active_conflict_result = evaluation_result['active_conflict_detection']
            if 'detected' in active_conflict_result:
                total_active_recognition_count += int(active_conflict_result['detected'])

            # 冲突提示响应率
            conflict_hint_result = evaluation_result['conflict_hint_response']
            if 'responded' in conflict_hint_result:
                total_conflict_hint_response_count += int(conflict_hint_result['responded'])

            # 模态依赖偏好值
            modality_preference_result = evaluation_result['modality_preference']
            if 'image_preference' in modality_preference_result and 'text_preference' in modality_preference_result:
                total_image_preference_count += modality_preference_result['image_preference']
                total_text_preference_count += modality_preference_result['text_preference']

            # 跨模态冲突敏感度
            cross_modal_result = evaluation_result['cross_modal_sensitivity']
            if 'sensitive' in cross_modal_result:
                total_cross_modal_sensitivity_count += int(cross_modal_result['sensitive'])

            total_samples += 1

            # 计算指标
            if total_samples == 0:
                active_recognition_rate = 0
                conflict_hint_response_rate = 0
                image_preference_rate = 0
                text_preference_rate = 0
                cross_modal_sensitivity_rate = 0
            else:
                active_recognition_rate = total_active_recognition_count / total_samples
                conflict_hint_response_rate = total_conflict_hint_response_count / total_samples
                image_preference_rate = total_image_preference_count / total_samples
                text_preference_rate = total_text_preference_count / total_samples
                cross_modal_sensitivity_rate = total_cross_modal_sensitivity_count / total_samples

            # 实时更新输出
            f.write(f"处理样本 PID: {pid}\n")
            f.write(f"主动冲突识别率: {active_recognition_rate:.2%}\n")
            f.write(f"冲突提示响应率: {conflict_hint_response_rate:.2%}\n")
            f.write(f"图像偏好值: {image_preference_rate:.2%}\n")
            f.write(f"文本偏好值: {text_preference_rate:.2%}\n")
            f.write(f"跨模态冲突敏感度: {cross_modal_sensitivity_rate:.2%}\n")
            f.write("\n")
            f.flush()  # 强制将缓冲区内容写入文件

        f.write("\n")

        return {
            "主动冲突识别率": active_recognition_rate,
            "冲突提示响应率": conflict_hint_response_rate,
            "图像偏好值": image_preference_rate,
            "文本偏好值": text_preference_rate,
            "跨模态冲突敏感度": cross_modal_sensitivity_rate
        }
    except Exception as e:
        logging.error(f"处理文件 {initial_file_path} 时出错: {e}")
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 o3 评估不同模型指标")
    parser.add_argument("--initial_folder", type=str, default="initial", help="initial 文件夹路径")
    parser.add_argument("--model_folder", type=str, required=True, help="要评估的模型文件夹路径")
    parser.add_argument("--model_name", type=str, required=True, help="被评估的模型名称")
    args = parser.parse_args()

    eval_model_name = "o3"
    # 修改 output_file 变量，添加被评估的模型名
    output_file = f"evaluation_results_{args.model_name}.txt"

    async def main():
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("评估开始\n\n")

        initial_jsonl_files = []
        for root, _, files in os.walk(args.initial_folder):
            for file in files:
                if file.endswith('.jsonl'):
                    initial_jsonl_files.append(os.path.join(root, file))

        logging.info(f"在 {args.initial_folder} 中找到 {len(initial_jsonl_files)} 个 JSONL 文件")

        for initial_file_path in initial_jsonl_files:
            initial_file_name = os.path.basename(initial_file_path)
            base_name, _ = os.path.splitext(initial_file_name)
            model_file_name = f"{base_name}_{args.model_name}.jsonl"
            model_file_path = os.path.join(args.model_folder, model_file_name)

            if not os.path.exists(model_file_path):
                logging.warning(f"未找到对应的模型回答文件: {model_file_path}")
                continue

            logging.info(f"开始处理文件: {initial_file_name}")
            try:
                await evaluate_file(initial_file_path, args.model_folder, args.model_name,
                                    eval_model_name, output_file)
                logging.info(f"完成处理文件: {initial_file_name}")
            except Exception as e:
                logging.error(f"处理文件 {initial_file_name} 时出错: {e}")

        print(f"评估结果已保存到 {output_file}")

    asyncio.run(main())
