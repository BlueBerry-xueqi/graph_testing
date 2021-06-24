#!/bin/bash
declare -a dataset=( "IMDB-BINARY"  "DD" "PROTEINS"  "MUTAG" "IMDB-MULTI" "COLLAB" )
declare -a models=( "gin" "gat" "gcn" "gmt" )

for i in $( seq 1 10 )
do
    for k in "${dataset[@]}"
    do
        for m in "${models[@]}"
        do
            [ -d repeat_exp/${k}/${m}/ ] || mkdir -p repeat_exp/${k}/${m}/
            python trainer_graph_classification.py  --type $m --data $k --model-string GMPool_G-SelfAtt-GMPool_I  --batch-size 10  --num-hidden 32 --num-heads 4  --lr-schedule   --cluster
            mv saved_model/${m}_${k}/performance.json repeat_exp/${k}/${m}/best_${i}.json
        done
    done
done