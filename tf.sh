#!/usr/bin/env bash

if [ $# -ne 4 ]; then
    echo "usage: img_num compress_num window_x window_y"
    exit 1
fi

cargo run --bin test_flow $1 $2 $3 $4