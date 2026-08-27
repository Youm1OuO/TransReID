#!/bin/bash

# 基础配置文件
CONFIG_FILE="/home/muyou/Projects/references/TransReID/configs/transformer_base_market1501_600_256x128.yml"


python train.py --config_file ${CONFIG_FILE} \
    MODEL.CLS_SEP 'True' \
    MODEL.CLS_GEN_TYPE 'dynamic' \
    MODEL.CLS_MLP_RATIO '4.0' \
    MODEL.USE_ROPE 'True' \
    SOLVER.OPTIMIZER_NAME 'SGD' \
    OUTPUT_DIR "./logs/Market_1501/Day1/e3_CLS_SEP=[True+dynamic+4.0]_USE_ROPE=[True]"
sleep 5


python train.py --config_file ${CONFIG_FILE} \
    MODEL.CLS_SEP 'True' \
    MODEL.CLS_GEN_TYPE 'static' \
    MODEL.CLS_MLP_RATIO '4.0' \
    MODEL.USE_ROPE 'True' \
    SOLVER.OPTIMIZER_NAME 'SGD' \
    OUTPUT_DIR "./logs/Market_1501/Day1/e4_CLS_SEP=[True+static+4.0]_USE_ROPE=[True]"
sleep 5




echo "所有消融实验全部跑完"



# 给脚本增加执行权限:
#    在终端中进入到当前脚本所在目录，然后执行以下命令
#    chmod +x run_ablation.sh

# 运行脚本:
#    在终端中进入到当前脚本所在目录，然后执行以下命令
#    ./run_ablation.sh