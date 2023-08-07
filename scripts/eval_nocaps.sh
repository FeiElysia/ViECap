SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
OTHER_ARGS=$3
EPOCH=$4
WEIGHT_PATH=checkpoints/$EXP_NAME/coco_prefix-00${EPOCH}.pt
NOCAPS_OUT_PATH=checkpoints/$EXP_NAME

TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs/${EXP_NAME}_EVAL
mkdir -p $LOG_FOLDER

NOCAPS_LOG_FILE="$LOG_FOLDER/NOCAPS_${TIME_START}.log"

python validation.py \
--device cuda:$DEVICE \
--clip_model ViT-B/32 \
--language_model gpt2 \
--continuous_prompt_length 10 \
--clip_project_length 10 \
--top_k 3 \
--threshold 0.2 \
--using_image_features \
--name_of_datasets nocaps \
--path_of_val_datasets ./annotations/nocaps/nocaps_corpus.json \
--name_of_entities_text vinvl_vgoi_entities \
--image_folder ./annotations/nocaps/ \
--prompt_ensemble \
--weight_path=$WEIGHT_PATH \
--out_path=$NOCAPS_OUT_PATH \
--using_hard_prompt \
--soft_prompt_first \
$OTHER_ARGS \
|& tee -a  ${NOCAPS_LOG_FILE}

echo "==========================NOCAPS IN-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/indomain*.json |& tee -a  ${NOCAPS_LOG_FILE}
echo "==========================NOCAPS NEAR-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/neardomain*.json |& tee -a  ${NOCAPS_LOG_FILE}
echo "==========================NOCAPS OUT-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/outdomain*.json |& tee -a  ${NOCAPS_LOG_FILE}
echo "==========================NOCAPS ALL-DOAMIN================================"
python evaluation/cocoeval.py --result_file_path  ${NOCAPS_OUT_PATH}/overall*.json |& tee -a  ${NOCAPS_LOG_FILE}

