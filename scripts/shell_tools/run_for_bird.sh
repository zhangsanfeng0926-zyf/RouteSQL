python scripts/python_tools/data_preprocess.py --data_type bird --data_dir ./dataset/bird

python scripts/python_tools/generate_question.py --data_type bird \
--split test --tokenizer gpt-3.5-turbo --prompt_repr SQL \
--selector_type EUCDISQUESTIONMASK --max_seq_len 4096 --k_shot 7 --example_type QA

python scripts/python_tools/ask_llm.py \
--openai_api_key $1 \
--model glm-4.5-air \
--question ./dataset/process/BIRD-TEST_SQL_7-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096/ \
--db_dir ./dataset/bird/databases

python scripts/python_tools/generate_question.py --data_type bird --split test --tokenizer gpt-3.5-turbo \
--prompt_repr SQL --max_seq_len 4096 --k_shot 7 --example_type QA --selector_type EUCDISMASKPRESKLSIMTHR \
--pre_test_result ./dataset/process/BIRD-TEST_SQL_7-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096/RESULTS_MODEL-glm-4.5-air.txt


python scripts/python_tools/ask_llm.py \
--openai_api_key $1 \
--model glm-4.5-air \
--question ./dataset/process/BIRD-TEST_SQL_7-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096/ \
--db_dir ./dataset/bird/databases

python to_bird_output.py --dail_output ./dataset/process/BIRD-TEST_SQL_7-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096/RESULTS_MODEL-glm-4.5-air.txt

cp ./dataset/process/BIRD-TEST_SQL_7-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096/RESULTS_MODEL-glm-4.5-air.json ./RESULTS_MODEL-glm-4.5-air.json

