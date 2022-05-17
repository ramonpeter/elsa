#!/bin/bash

python datasets/get_lhc_data.py

cd datasets/toy
python generate_8gaussians.py
python generate_circles.py
python generate_pinwheel.py