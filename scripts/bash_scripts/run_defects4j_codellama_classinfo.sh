python3 ../llama_pred.py \
    --base_model_path codellama/CodeLlama-7b-hf \
    --lora_path /projects/ksun3/dwu25/trained_models/codellama_classinfo_finetune_lora \
    --data_path /projects/ksun3/dwu25/apr_datasets_processing/classinfo_defects4j_comparison/datasets/defects4j_classinfo \
    --output_path /projects/ksun3/dwu25/datasets/defects4j_codellama_classinfo_gen \
    --is_lora False \
    --max_length 2048 \
    --max_new_tokens 256 \
    --do_sample True \
    --only_do_beam True \
    --only_do_topp False \
    --only_do_topk False \
    --only_do_temp False \
    --num_beams 10 \
    --temperature 0.8 \
    --top_k 0 \
    --top_p 0.95 \
    --request_num 10