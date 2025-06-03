
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import json
import time
import requests
import regex as re
import tiktoken
from typing import List, Dict, Any
import logging


class APIModel:

    def __init__(self, api_key, api_url, model_name, model_path, model_name_embd) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.model_name_embd = model_name_embd
        self.model_path =model_path
        #  "/home/kunzhu/models/llama/Llama-3.1-8B-Instruct"
        # model_name = "Llama-3.1-8B-Instruct"
        self.api_model = OpenAI(api_key=api_key, base_url=api_url)

        if "llama" in model_name:
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False
            )
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
            ).to("cuda").eval()
        else:
            print("No Loading Llama")
        self.input_token_count = 0
        self.output_token_count = 0
        self.embedding_token_count = 0

    def __req_llama(self, system_prompt, user_message):    
        message = []
        message.append({"role": "system", "content":system_prompt})
        message.append({"role":"user", "content":user_message})
        # print(user_message)
        # print(message)
        input_ids = self.llama_tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(self.llama_model.device)
        seq_len = input_ids.shape[-1]
        self.input_token_count += seq_len

        attention_mask = torch.ones_like(input_ids).to(self.llama_model.device)
        # print(input_ids.shape, attention_mask.shape)
        output = self.llama_model.generate(input_ids, max_new_tokens=25000, use_cache=True,attention_mask=attention_mask,pad_token_id=self.llama_tokenizer.eos_token_id)
        self.output_token_count += len(output[0])-seq_len
        output = self.llama_tokenizer.decode(output[0, seq_len:], skip_special_tokens=True).strip()
        # print("-"*20)
        # print(output)
        # print("-"*20)
        match = re.search(r"[\s\S]*?({[\s\S]*?})[\s\S]*?", output)
        if match:
            json_str = match.group(1)
            try:
                data = json.loads(json_str)
                # print("提取成功，内容如下：")
                # print(json.dumps(data, indent=2))
                return data
            except json.JSONDecodeError as e:
                print("JSON 解码失败:", e)
                print(json_str)
        else:
            print("未找到 JSON 内容")
            print(output)

        return None
    
    def __req_api(self, model_name, system_prompt, user_message, temperature=0.1, max_try = 5):
        # url = f"{self.__api_url}"
        # pay_load_dict = {"model": f"{self.model}","messages": [{
        #         "role": "user",
        #         "temperature":temperature,
        #         "content": f"{text}"}]}
        # payload = json.dumps(pay_load_dict)
        # headers = {
        # 'Accept': 'application/json',
        # 'Authorization': f'Bearer {self.__api_key}',
        # 'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        # 'Content-Type': 'application/json'
        # }
        for _ in range(max_try):
            try:
            # print(11111)
            # response = requests.request("POST", url, headers=headers, data=payload)
            # print("response:",response,json.loads(response.text))
            # return json.loads(response.text)['choices'][0]['message']['content']

                response = self.api_model.chat.completions.create(
                    model=model_name,
                    response_format={ "type": "json_object" },
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content":  user_message}
                    ],
                    temperature=temperature
                )
                print(response.choices[0].message.content)
                self.input_token_count += response.usage.prompt_tokens
                self.output_token_count += response.usage.completion_tokens
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(e)
                pass
            time.sleep(0.2)
        return None
    
    def __req_claude(self, system_prompt: str, user_message: str, model: str = "aws-claude3.7", 
                     temperature: float = 1.0, top_p: float = 0.95, max_tokens: int = 10240) -> Dict:
        url = "https://pre.in2x.com/nlp-router/stream"
        headers = {"Content-Type": "application/json"}
        messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content":  user_message}
                ]
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=90
                )
                response.raise_for_status()
                
                response_json = response.json()
                content = response_json['choices'][0]['message']['content']

    
                # 计算 Token 使用量
                usage = response_json.get('usage', {})
                self.input_token_count +=  usage.get('input_tokens', None)
                self.output_token_count += usage.get('output_tokens', None)

                return json.loads(content)
            
            except requests.exceptions.RequestException as e:
                error_msg = f"API请求失败: {str(e)}"
                logging.warning(f"{error_msg}，第{attempt+1}/{max_retries}次尝试")
                
                if attempt == max_retries - 1:
                    return None
                time.sleep(2 * (attempt + 1))



    def req(self, system_prompt, user_message, temperature=0.1, max_try = 5):
        if "llama" in self.model_name:
            return self.__req_llama(system_prompt, user_message)
        elif "gpt" in self.model_name:
            return self.__req_api(self.model_name, system_prompt, user_message, temperature=temperature, max_try=max_try)
        elif "claude" in self.model_name:
            return self.__req_claude(system_prompt, user_message, self.model_name)
        
        else:
            raise Exception("No Match Model")


    def embedding_api(self, data_for_embedding, max_try = 5):
        
        encoding = tiktoken.encoding_for_model(self.model_name_embd)
        if isinstance(data_for_embedding, str):
            tokens = encoding.encode(data_for_embedding)
            self.embedding_token_count += len(tokens)
        elif isinstance(data_for_embedding, list):
            self.embedding_token_count += sum(len(encoding.encode(item)) for item in data_for_embedding)
        else:
            raise ValueError("data_for_embedding 必须是 str 或 list[str]")
        
        for _ in range(max_try):
            try:
                response = self.api_model.embeddings.create(
                    input=data_for_embedding,
                    model=self.model_name_embd
                )
                # print(11111111111)
                return response.data
            except Exception as e:
                print(e)
                pass
            time.sleep(0.2)
        return None

