import torch 
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoModelForVision2Seq
from openai import OpenAI
import base64
from PIL import Image
from io import BytesIO
import requests
from torchvision import transforms
from modelscope import AutoProcessor, AutoModelForImageTextToText


def build_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def dynamic_preprocess(image, image_size, use_thumbnail, max_num):
    return [image]

class VLM:
    def __init__(self, model_name, model_args):
        self.model_name = model_name
        self.model_args = model_args
        cache_dir = "/autodl-tmp/huggingface_cache"  # 定义缓存目录
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        device_map = "auto" if self.device == "cuda" else None
        model_cache ="/root/autodl-fs/models/CohereForAI/aya-vision-32b"
        # 闭源模型统一使用青云聚代理 API（包括 gpt-4o、claude、gemini、o3）
        if any(x in model_name.lower() for x in ["gpt-4o", "claude", "gemini", "o3"]):
            self.client = OpenAI(
                base_url="https://api.qingyuntop.top/v1",
                api_key="sk-32QMw0xzJWlYYCePHAuOCT4XvtiM6VTfDdBQip6Qay40IHBW"
            )
        # # 使用魔塔 API 的 Qwen 系列模型
        elif model_name == "Qwen2.5-VL-32B-Instruct":
            self.client = OpenAI(
                base_url='https://api-inference.modelscope.cn/v1/',
                api_key='6d13ee68-7474-4dab-909e-d5b0806ebd52'  # ModelScope Token
            )
        elif model_name == "Qwen2.5-VL-7B-Instruct":
            self.client = OpenAI(
                base_url='https://api-inference.modelscope.cn/v1/',
                api_key='95fa1359-3cec-439b-a339-109e7d97d900'  # ModelScope Token
            )
        # 加载 meta-llama/Llama-3.2-11B-Vision-Instruct 模型
        elif model_name == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                cache_dir=cache_dir
            ).to(self.device)
            self.model.eval()

        elif model_name == "OpenGVLab/InternVL3-38B-Instruct":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map,
                cache_dir=cache_dir,
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=False,
                cache_dir=cache_dir
            )
        elif "Grok" in model_name:
            self.client = OpenAI(
                base_url="https://api2.aigcbest.top/v1",
                api_key="sk-ZR4ButF7gUhfaHd0p2aLRo3Y3HN84jcgtLZg3fTrN9JalmRF"
            )
        # 添加 CohereForAI/aya-vision-8b 模型加载逻辑
        elif model_name == "CohereForAI/aya-vision-8b":
            # 设置 use_fast 参数，根据需求选择 True 或 False
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch.float16
            )
            self.device = next(self.model.parameters()).device
        
        elif model_name == "CohereForAI/aya-vision-32b":
            # 设置 use_fast 参数，根据需求选择 True 或 False
            self.processor = AutoProcessor.from_pretrained(model_cache, use_fast=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_cache, 
                device_map="balanced_low_0",  # 模型在多张 GPU 卡间均匀分配
                torch_dtype=torch.float16
            )
            self.device = next(self.model.parameters()).device
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_response(self, messages):
        """
        根据包含文本和图片信息的消息列表获取模型响应
        :param messages: 包含文本和图片信息的消息列表
        :return: 模型的响应内容
        """
        try:
            model_lower = self.model_name.lower()

            # 调用 gpt-4o、claude、gemini、o3（统一 OpenAI SDK + 青云聚 API）
            if any(x in model_lower for x in ["gpt-4o", "claude", "gemini", "o3"]):
                model_map = {
                    "gpt-4o": "gpt-4o",
                    "claude": "claude-sonnet-4-20250514",
                    "gemini": "gemini-2.5-pro-exp-03-25",
                    "o3": "o3"
                }
                model = next((v for k, v in model_map.items() if k in model_lower), None)
                if model:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0  
                    )
                    return response.choices[0].message.content
            #调用 Grok
            elif "grok" in model_lower:
                response = self.client.chat.completions.create(
                    model="grok-3",  # 根据实际情况调整模型名称
                    messages=messages,
                    temperature=0.0  
                )
                return response.choices[0].message.content
            # 使用魔塔 API 的 Qwen 系列模型
            elif self.model_name == "Qwen2.5-VL-32B-Instruct":
                try:
                    response = self.client.chat.completions.create(
                        model='Qwen/Qwen2.5-VL-32B-Instruct',
                        messages=messages,
                        stream=False,  # 暂时关闭流式响应
                        temperature=0.3 
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"调用魔塔 API 失败，模型 {self.model_name}: {e}")
                    return ""
            elif self.model_name == "Qwen2.5-VL-7B-Instruct":
                try:
                    response = self.client.chat.completions.create(
                        model='Qwen/Qwen2.5-VL-7B-Instruct',
                        messages=messages,
                        stream=False,  # 暂时关闭流式响应
                        temperature=0.3  
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"调用魔塔 API 失败，模型 {self.model_name}: {e}")
                    return ""
            # 调用 OpenGVLab/InternVL3-38B-Instruct 模型
            elif self.model_name == "OpenGVLab/InternVL3-38B-Instruct":
                question = messages[0]["content"][0]["text"]
                if "image_url" in messages[0]["content"][1]:
                    image_url = messages[0]["content"][1]["image_url"]["url"]
                    if image_url.startswith('data:image'):
                        base64_image = image_url.split(',')[1]
                        img_bytes = base64.b64decode(base64_image)
                        image_file = BytesIO(img_bytes)
                    else:
                        try:
                            response = requests.get(image_url, stream=True)
                            response.raise_for_status()
                            image_file = BytesIO(response.content)
                        except Exception as e:
                            print(f"获取图片失败: {e}")
                            return ""
                else:
                    print("未提供有效的图片信息")
                    return ""

                pixel_values = self.load_image(image_file)
                # 将 pixel_values 数据类型转换为 torch.float16
                if self.device == "cuda":
                    pixel_values = pixel_values.to(torch.float16).to(self.device)
                else:
                    pixel_values = pixel_values.to(self.device)

                # 检查 question 是否为空
                if not question.strip():
                    print("输入问题为空，请提供有效的问题。")
                    return ""

                generation_config = self.model_args.__dict__
                # 确保 max_new_tokens 被正确设置
                if "max_new_tokens" not in generation_config:
                    generation_config["max_new_tokens"] = 512
                # 确保 temperature 为 0.0
                generation_config["temperature"] = 0.0  

                with torch.no_grad():
                    response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
                return response
           
            # 添加 CohereForAI/aya-vision-8b 模型推理逻辑
            elif self.model_name == "CohereForAI/aya-vision-8b":
                inputs = self.processor.apply_chat_template(
                    messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                ).to(self.device)

                gen_tokens = self.model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=True, 
                    temperature=0.3,#默认
                )

                return self.processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            elif self.model_name == "CohereForAI/aya-vision-32b":
                inputs = self.processor.apply_chat_template(
                    messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
                ).to(self.device)

                gen_tokens = self.model.genera
                    **inputs, 
                    max_new_tokens=512, 
                    do_sample=True, 
                    temperature=0.3,#默认
                )

                return self.processor.tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            print(f"获取 {self.model_name} 模型响应时出错: {e}")
            return ""
