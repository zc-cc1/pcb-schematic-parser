#!/usr/bin/env python3
"""
赛题六：PCB电路原理图解析 - 简化版
使用更宽容的解析策略
"""
import os
import sys
import json
import base64
import argparse
from pathlib import Path
from datetime import datetime, timezone
from openai import OpenAI


class SimplePCBParser:
    """简化的PCB解析器"""

    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = "qwen-vl-plus"
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0

    def parse_image(self, image_path):
        """解析单张图片"""
        print(f"[Agent] Processing: {image_path}")

        # 读取图片
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # 简化的提示词
        prompt = """分析这张PCB电路原理图，识别元件、引脚和连接。

输出JSON格式（不要添加任何其他文字）：
{
  "components": {"元件ID": {"bbox": [x1,y1,x2,y2], "type": "类型", "center": [x,y]}},
  "pins": {"引脚ID": {"component": "元件ID", "position": [x,y], "pin_number": "编号"}},
  "nets": {"网络ID": {"pins": ["引脚列表"], "wire_segments": []}}
}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0
            )

            content = response.choices[0].message.content

            # 统计token
            self.total_tokens += response.usage.total_tokens
            self.prompt_tokens += response.usage.prompt_tokens
            self.completion_tokens += response.usage.completion_tokens
            self.api_calls += 1

            print(f"[Agent] API call successful, tokens: {response.usage.total_tokens}")

            # 尝试解析
            result = self._parse_content(content)
            return result

        except Exception as e:
            print(f"[Agent] Error: {e}")
            return self._empty_result()

    def _parse_content(self, content):
        """宽容的内容解析"""
        import re
        import ast

        # 提取JSON块
        start = content.find('{')
        end = content.rfind('}')

        if start == -1 or end == -1:
            print("[Agent] No JSON found")
            return self._empty_result()

        json_str = content[start:end+1]

        # 方法1：直接解析JSON
        try:
            return json.loads(json_str)
        except:
            pass

        # 方法2：修复常见错误后再解析
        try:
            # 移除尾部逗号
            fixed = re.sub(r',\s*}', '}', json_str)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except:
            pass

        # 方法3：使用ast.literal_eval（更宽容）
        try:
            # 将JSON转换为Python字面量格式
            py_str = json_str.replace('null', 'None').replace('true', 'True').replace('false', 'False')
            return ast.literal_eval(py_str)
        except:
            pass

        print("[Agent] All parsing methods failed")
        return self._empty_result()

    def _empty_result(self):
        """空结果"""
        return {"components": {}, "pins": {}, "nets": {}}

    def get_token_summary(self):
        """获取token统计"""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "api_calls": self.api_calls
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', required=True)
    parser.add_argument('-output', required=True)
    args = parser.parse_args()

    # 获取API密钥（方式1：代码中硬编码，带fallback）
    api_key = os.environ.get('DASHSCOPE_API_KEY', 'sk-你的API密钥')

    print("[Agent] ========================================")
    print("[Agent] Simple PCB Parser (Method 1: Hardcoded API Key)")
    print("[Agent] ========================================")
    print(f"[Agent] Input: {args.input}")
    print(f"[Agent] Output: {args.output}")

    # 创建输出目录
    output_path = Path(args.output)
    final_results = output_path / "final_results"
    final_results.mkdir(parents=True, exist_ok=True)

    # 初始化解析器
    pcb_parser = SimplePCBParser(api_key)

    # 处理所有PNG图片
    input_path = Path(args.input)
    images = sorted(input_path.glob("*.png"))

    print(f"[Agent] Found {len(images)} images")

    for img in images:
        result = pcb_parser.parse_image(img)

        # 保存结果
        output_file = final_results / f"{img.stem}_Pred.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"[Agent] Saved: {output_file}")

    # 保存decision_log
    token_summary = pcb_parser.get_token_summary()
    decision_log = {
        "method": "qwen-vl-plus",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processed_cases": len(images),
        "token_usage": token_summary
    }

    with open(final_results / "decision_log.json", 'w') as f:
        json.dump(decision_log, f, indent=2)

    print(f"[Agent] Total tokens: {token_summary['total_tokens']}")
    print("[Agent] ========================================")
    print("[Agent] Completed")
    print("[Agent] ========================================")


if __name__ == "__main__":
    main()
