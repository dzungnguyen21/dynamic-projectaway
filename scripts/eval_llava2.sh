#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/path/to/your/llava2

MODEL_NAME="llava2"
MODEL_VERSION="llava-hf/llava-1.5-7b-hf"
BATCH_SIZE=4
SEED=242

# Intervention method: 'marine' (original CFG) or 'dynamic_projectaway' (DPA)
METHOD=dynamic_projectaway

guidance_strength_lst=(0.0 0.7)
TYPE=repro

# ---- Dynamic-PROJECTAWAY hyper-parameters (only used when METHOD=dynamic_projectaway) ----
HGAI_LAYERS="5-18"          # Visual Information Enrichment zone
PA_LAYERS="19-26"           # Semantic Refinement zone
DILUTION_THRESHOLD=0.10     # VAR proxy floor; below this triggers intervention
HGAI_AMPLIFY=2.0            # Consensus attention amplification factor
SINK_THRESHOLD=2.0          # Sink detection multiplier (attn > SINK_THRESHOLD * mean)

BENCHMARK=chair
if [ $BENCHMARK == "chair" ]; then
    QUESTION_FILE_ls=(chair_coco_detr_th0.95_ram_th0.68.json)
elif [ $BENCHMARK == "pope" ]; then
    QUESTION_FILE_ls=(pope_coco_detr_th0.95_ram_th0.68.json)
fi

OUTPUT_DIR=./output/${MODEL_NAME}/answers/answer_${TYPE}_${BENCHMARK}

#### Generate answers ####
for guidance_strength in "${guidance_strength_lst[@]}"; do
    for QUESTION_FILE in "${QUESTION_FILE_ls[@]}"; do
        echo "Running $MODEL_VERSION inference | method=$METHOD | guidance_strength=$guidance_strength | seed=$SEED | batch_size=$BATCH_SIZE"
        python ./marine/generate_${MODEL_NAME}.py \
            --question_file $QUESTION_FILE \
            --guidance_strength $guidance_strength \
            --answer_path $OUTPUT_DIR \
            --model_path $MODEL_VERSION \
            --seed $SEED \
            --batch_size $BATCH_SIZE \
            --image_folder ./data/coco/val2014 \
            --temperature 0.6 \
            --top_p 0.9 \
            --max_new_tokens 64 \
            --method $METHOD \
            --hgai_layers $HGAI_LAYERS \
            --pa_layers $PA_LAYERS \
            --dilution_threshold $DILUTION_THRESHOLD \
            --hgai_amplify_factor $HGAI_AMPLIFY \
            --sink_threshold $SINK_THRESHOLD
    done
done


#### EVALUATE ####
if [ $BENCHMARK == "chair" ]; then

    #### CHAIR EVALUATION ####
    echo "Running $MODEL_VERSION CHAIR metrics with seed = $SEED, batch_size = $BATCH_SIZE"

    python ./eval/format.py \
        --answer_dir $OUTPUT_DIR

    python ./eval/eval_chair.py \
        --eval_dir $OUTPUT_DIR \
        --save_path $OUTPUT_DIR/eval \

elif [ $BENCHMARK == "pope" ]; then

    #### POPE EVALUATION ####
    for QUESTION_FILE in "${QUESTION_FILE_ls[@]}"; do

        echo "Running $MODEL_VERSION pope evaluation with seed = $SEED, batch_size = $BATCH_SIZE"

        python ./eval/format.py \
            --answer_dir $OUTPUT_DIR

        python ./eval/eval_pope.py \
            --eval_dir $OUTPUT_DIR \
            --save_dir $OUTPUT_DIR/eval \
            --label_file $QUESTION_FILE \

    done
fi
