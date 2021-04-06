#!/bin/bash

FILE=./experiments/siamrpn_r50_l234_dwxcorr/model.pth
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist. Starting download..."
    gdown https://drive.google.com/uc?id=1-tEtYQdT1G9kn8HsqKNDHVqjE16F8YQH
    mv model.pth ./experiments/siamrpn_r50_l234_dwxcorr
fi
export PYTHONPATH=./pysot:$PYTHONPATH
python spawn.py \
        --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
        --snapshot experiments/siamrpn_r50_l234_dwxcorr/model.pth
