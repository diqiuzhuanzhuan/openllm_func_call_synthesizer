# 模型训练
# mcp_train_1030 整个文件拷贝到  docker里的目录

# cp -rf /data0/work/SusieSu/project/openllm_func_call_synthesizer/data/function_call_for_train_1112/mcp_data_1112_for_train   /data0/work/SusieSu/project/workspace/llama/LLaMA-Factory/data/mcp_train_data
# cd /data0/work/SusieSu/project/workspace/llama/LLaMA-Factory/data/mcp_train_data/mcp_data_1103_for_train
# cp -f mcp_train.json mcp_dev.json ../../

# python generate_functio_call_format_data.py
# python data_process_for_train_pipline.py

cp -rf /data0/work/SusieSu/project/openllm_datas_and_temp_codes/data_1119/mcp_data_1119_for_train  /data0/work/SusieSu/project/workspace/LLaMA-Factory-main/data/mcp_train_data/
cd  /data0/work/SusieSu/project/workspace/LLaMA-Factory-main/data/mcp_train_data/mcp_data_1119_for_train
cp -f mcp_train.json mcp_dev.json ../../


# cp -rf /data0/work/SusieSu/project/openllm_datas/processed_data/mcp_data_1117_for_train   /data0/work/SusieSu/project/workspace/llama/LLaMA-Factory/data/mcp_train_data
# cd /data0/work/SusieSu/project/workspace/llama/LLaMA-Factory/data/mcp_train_data/mcp_data_1117_for_train
# cp -f mcp_train.json mcp_dev.json ../../