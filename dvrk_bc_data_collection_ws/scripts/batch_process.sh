#!/bin/bash
# Process all rosbags in a directory to LeRobot dataset

RAW_DIR="$1"
OUTPUT_DIR="$2"
FPS="${3:-30}"  # Optional third argument, defaults to 30

if [ -z "$RAW_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <raw_data_dir> <output_lerobot_dir> [fps]"
    echo ""
    echo "Arguments:"
    echo "  raw_data_dir      : Directory containing .bag files"
    echo "  output_lerobot_dir: Output directory for LeRobot dataset"
    echo "  fps               : Target framerate (default: 30)"
    echo ""
    echo "Example:"
    echo "  $0 data/raw_20231215_120000/ data/lerobot/ 30"
    exit 1
fi

# Check if raw dir exists
if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: Raw data directory does not exist: $RAW_DIR"
    exit 1
fi

# Find all .bag files
bags=($(find "$RAW_DIR" -name "*.bag" | sort))

if [ ${#bags[@]} -eq 0 ]; then
    echo "ERROR: No .bag files found in $RAW_DIR"
    exit 1
fi

echo "Found ${#bags[@]} rosbag files"
echo "Target FPS: $FPS"
echo ""

# Process each bag
for i in "${!bags[@]}"; do
    bag="${bags[$i]}"
    echo ""
    echo "Processing bag $((i+1))/${#bags[@]}: $bag"

    python3 scripts/convert_rosbag_to_lerobot.py \
        --bag "$bag" \
        --output "$OUTPUT_DIR" \
        --episode-idx "$i" \
        --fps "$FPS"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to process $bag"
        exit 1
    fi
done

echo ""
echo "✓ All episodes processed successfully"
echo ""
echo "Finalizing LeRobot dataset (generating info.json and episodes.jsonl)..."

python3 scripts/convert_rosbag_to_lerobot.py \
    --output "$OUTPUT_DIR" \
    --fps "$FPS" \
    --finalize

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to finalize dataset"
    exit 1
fi

echo ""
echo "✓ Dataset finalized and ready for LeRobot compatibility"
echo "LeRobot dataset: $OUTPUT_DIR"
