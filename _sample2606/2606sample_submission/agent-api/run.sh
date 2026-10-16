#!/bin/bash
# Sample submission entry point for 赛题六

# 解析参数
INPUT_DIR=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# 验证参数
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 -input <input_dir> -output <output_dir>"
    exit 1
fi

echo "[Sample Agent] Starting PCB schematic parsing..."
echo "[Sample Agent] Input directory: $INPUT_DIR"
echo "[Sample Agent] Output directory: $OUTPUT_DIR"

# 创建输出目录
mkdir -p "$OUTPUT_DIR/final_results"

# 运行Python agent
python3 /submission/agent.py \
    -input "$INPUT_DIR" \
    -output "$OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[Sample Agent] Completed successfully"
else
    echo "[Sample Agent] Failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
