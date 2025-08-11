#!/bin/bash

grep "Test:" | awk '{ printf "%20s acc: %s IoU: %s\n", $9, $16, $19 }' | sort -k 5