#!/bin/sh

OUT_DIR="/mnt/c/Users/User/Desktop/pcd_results"
# NAMES="10881_Gim3d"
# NAMES="11775_Gim3dFr100 07188_Gim3dFr200 06738_Gim3d 11444_Gim3dFr200 07058_Gim3dFr100 07484_Gim3d 10550_Gim3d 11691_Gim3d 07156_Gim3d"
# NAMES="10550_Gim3dFr100"
NAMES="10075_7356"
DATASET="close"

for NAME in $NAMES;
do

    mkdir -p "$OUT_DIR/$NAME";

    # layer 0
    for strategy in $(seq 2 5);
    do
        python tools/view_results.py \
            -d $DATASET \
            -n pt-t$strategy-aug \
            --layer 0 \
            --colors gray yellow \
            --name $NAME \
            --save_path "$OUT_DIR/$NAME/pt-t$strategy-aug-layer0.ply" \
            --noshow;
    done

    # t2
    python tools/view_results.py \
        -d $DATASET \
        -n pt-t2-aug \
        --layer 1 \
        --colors gray red \
        --name $NAME \
        --save_path "$OUT_DIR/$NAME/pt-t2-aug-layer1.ply" \
        --noshow;

    python tools/view_results.py \
        -d $DATASET \
        -n pt-t2-aug \
        --layer 2 \
        --colors gray blue \
        --name $NAME \
        --save_path "$OUT_DIR/$NAME/pt-t2-aug-layer2.ply" \
        --noshow;
    #####

    # t3
    python tools/view_results.py \
        -d $DATASET \
        -n pt-t3-aug \
        --layer 1 \
        --colors gray red blue \
        --name $NAME \
        --save_path "$OUT_DIR/$NAME/pt-t3-aug-layer1.ply" \
        --noshow;

    python tools/view_results.py \
        -d $DATASET \
        -n pt-t3-aug \
        --layer 2 \
        --colors gray blue \
        --name $NAME \
        --save_path "$OUT_DIR/$NAME/pt-t3-aug-layer2.ply" \
        --noshow;
    #####

    # t4
    python tools/view_results.py \
        -d $DATASET \
        -n pt-t4-aug \
        --layer 1 \
        --colors gray red orange yellow \
        --name $NAME \
        --save_path "$OUT_DIR/$NAME/pt-t4-aug-layer1.ply" \
        --noshow;

    python tools/view_results.py \
        -d $DATASET \
        -n pt-t4-aug \
        --layer 2 \
        --colors gray blue light_blue green \
        --name $NAME \
        --save_path "$OUT_DIR/$NAME/pt-t4-aug-layer2.ply" \
        --noshow;
    #####

    # t5
    python tools/view_results.py \
        -d $DATASET \
        -n pt-t5-aug \
        --layer 1 \
        --colors gray orange light_blue blue red yellow green \
        --name $NAME \
        --save_path "$OUT_DIR/$NAME/pt-t5-aug-layer1.ply" \
        --noshow;

    python tools/view_results.py \
        -d $DATASET \
        -n pt-t5-aug \
        --layer 2 \
        --colors gray green light_blue blue \
        --name $NAME \
        --save_path "$OUT_DIR/$NAME/pt-t5-aug-layer2.ply" \
        --noshow;
    #####

done
