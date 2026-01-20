cd retrieval_lm


export PYTHONPATH="$(pwd):$PYTHONPATH"

python ./inference.py \
--model_name_or_path \
"zorowin123/rq_rag_llama2_7B" \
--input_file \
"/home/carson/research_projects/rq-rag/data/2wiki/dev_1.json" \
--max_new_tokens \
100 \
--output_path \
"/home/carson/research_projects/rq-rag/output/dev_1" \
--ndocs \
3 \
--use_search_engine \
--use_hf \
--task \
2wikimultihopqa \
--tree_decode \
--oracle \
--max_depth \
2 \
--search_engine_type \
openai_embed \
--expand_on_tokens \
[S_Rewritten_Query] \
[S_Decomposed_Query] \
[S_Disambiguated_Query] \
[A_Response] \
--overwrite_output_dir \