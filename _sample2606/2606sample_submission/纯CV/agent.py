#!/usr/bin/env python3
"""
Dockerfile提交方式的测试脚本
返回空结果，用于验证Dockerfile构建和评测流程
"""
import json
import argparse
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', required=True, help='Input directory')
    parser.add_argument('-output', required=True, help='Output directory')
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    final_results = output_dir / 'final_results'
    final_results.mkdir(parents=True, exist_ok=True)

    print(f"[Dockerfile Test] Input: {input_dir}")
    print(f"[Dockerfile Test] Output: {output_dir}")

    # 处理所有PNG
    png_files = list(input_dir.glob('*.png'))
    print(f"[Dockerfile Test] Found {len(png_files)} images")

    for image_file in png_files:
        case_name = image_file.stem

        # 空结果
        result = {
            "components": {},
            "pins": {},
            "nets": {}
        }

        output_file = final_results / f"{case_name}_Pred.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[Dockerfile Test] Processed: {case_name}")

    # 生成decision_log
    decision_log = {
        "method": "dockerfile_test",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "processed_cases": len(png_files),
        "token_usage": {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "api_calls": 0
        }
    }

    with open(output_dir / 'decision_log.json', 'w') as f:
        json.dump(decision_log, f, indent=2)

    print(f"[Dockerfile Test] Completed {len(png_files)} cases")

if __name__ == "__main__":
    main()
