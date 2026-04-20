echo "data_preprocess"
python scripts/python_tools/data_preprocess.py

echo "generate question with EUCDISQUESTIONMASK"
python scripts/python_tools/generate_question.py \
--data_type spider \
--split test \
--tokenizer gpt-3.5-turbo \
--max_seq_len 10000 \
--prompt_repr SQL \
--k_shot 9 \
--example_type QA \
--selector_type  EUCDISQUESTIONMASK

echo "generate SQL by glm-4.5-air for EEUCDISQUESTIONMASK as the pre-generated SQL query"
python scripts/python_tools/ask_llm.py \
--openai_api_key $1  \
--model glm-4.5-air \
--question ./dataset/process/SPIDER-TEST_SQL_9-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-10000/ \
--n 5 \
--db_dir ./dataset/spider/database \
--temperature 1.0

echo "generate question with EUCDISMASKPRESKLSIMTHR"
python scripts/python_tools/generate_question.py \
--data_type spider \
--split test \
--tokenizer gpt-3.5-turbo \
--max_seq_len 10000 \
--selector_type EUCDISMASKPRESKLSIMTHR \
--pre_test_result ./dataset/process/SPIDER-TEST_SQL_9-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-10000/RESULTS_MODEL-glm-4.5-air.txt \
--prompt_repr SQL \
--k_shot 9 \
--example_type QA

echo "generate SQL by glm-4.5-air for EUCDISMASKPRESKLSIMTHR"
python scripts/python_tools/ask_llm.py \
--openai_api_key $1  \
--model glm-4.5-air \
--question ./dataset/process/SPIDER-TEST_SQL_9-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-10000/ \
--n 5 \
--db_dir ./dataset/spider/database \
--temperature 1.0