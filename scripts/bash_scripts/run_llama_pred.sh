python3 ../llama_pred.py \
    --base_model_path codellama/CodeLlama-7b-hf \
    --lora_path /projects/ksun3/dwu25/repairllama/repairllama-lora \
    --data_path /projects/ksun3/dwu25/repairllama/results/defects4j/repairllama/lora \
    --test_file RepairLLaMA_defects4j_f2f_bugs_results_ir4_or2.jsonl \
    --output_file defects4j_with_buggyline_repairllama_gen.jsonl \
    --is_lora True \
    --max_length 1024 \
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
    --request_num 10 \
