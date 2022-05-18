#!/bin/bash

cd datasets/lhc
python get_lhc_data.py

cd ../toy
python generate_8gaussians.py
python generate_circles.py
python generate_pinwheel.py