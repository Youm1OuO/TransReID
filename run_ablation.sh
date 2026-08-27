#!/bin/bash

# 基础配置文件
CONFIG_FILE="/home/muyou/Projects/references/TransReID/configs/transformer_base_market1501_600_256x128.yml"


python train.py --config_file ${CONFIG_FILE} \
    MODEL.JPM 'False' \
    MODEL.SIE_CAMERA 'False' \
    MODEL.SIE_VIEW 'False' \
    MODEL.SIE_COE '3.0' \
    MODEL.CLS_SEP 'True' \
    MODEL.CLS_GEN_TYPE 'dynamic' \
    MODEL.CLS_MLP_RATIO '4.0' \
    MODEL.USE_ROPE 'True' \
    MODEL.ABS_POS_MODE 'cross_attn' \
    SOLVER.OPTIMIZER_NAME 'SGD' \
    SOLVER.MAX_EPOCHS '300' \
    SOLVER.EVAL_PERIOD '20' \
    OUTPUT_DIR "./logs/Market_1501/Day1/e6_300epoch_CLS_SEP=[True+dynamic+4.0]_USE_ROPE=[True]_ABS_POS_MODE=[cross_attn]"
sleep 5



echo "所有消融实验全部跑完"



# 给脚本增加执行权限:
#    在终端中进入到当前脚本所在目录，然后执行以下命令
#    chmod +x run_ablation.sh

# 运行脚本:
#    在终端中进入到当前脚本所在目录，然后执行以下命令
#    ./run_ablation.sh