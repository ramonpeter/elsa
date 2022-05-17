""" Get lhc datasets """

import os
import wget

# Make ttbar DIR
DIR = "datasets/wp_nj"
PATH = os.path.abspath(DIR)
if not os.path.exists(PATH):
    os.makedirs(PATH)

URLS = [
    "https://www.dropbox.com/s/bs8go6jqch5nuqy/wp_2j.h5?dl=1",
    "https://www.dropbox.com/s/vtvrryw4rjat5pz/wp_3j.h5?dl=1",
    "https://www.dropbox.com/s/1fc7s4ucqyhbw7k/wp_4j.h5?dl=1",
    "https://www.dropbox.com/s/q7dwbox40sky10f/README.txt?dl=1",
]

NAMES = ["wp_2j.h5", "wp_3j.h5", "wp_4j.h5", "README.txt"]

for url, name in zip(URLS, NAMES):
    wget.download(url, f"{DIR}/{name}")