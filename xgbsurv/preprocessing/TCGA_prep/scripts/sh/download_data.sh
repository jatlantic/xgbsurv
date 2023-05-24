#!/bin/bash

mkdir -p ./data
mkdir -p ./data/raw

# Download everything except CNV from PANCANATLAS
for id in "3586c0da-64d0-4b74-a449-5ff4d9136611" \
    "1b5f413e-a8d1-4d10-92eb-7c4ae739ed81" \
    "0fc78496-818b-4896-bd83-52db1f533c5c"; do
    wget --content-disposition http://api.gdc.cancer.gov/data/${id} -P ./data/raw/
    sleep 60
done
