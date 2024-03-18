import subprocess
import os
import sys

SRC = 'gmei@diclubclsin.fbk.eu:/storage2/TEV/datasets/Matterport/matterport_2d/test'
DEST = '/data/disk1/data/Matterport/matterport_2d/'
KEY = sys.argv[1]

subprocess.call(f'rsync -zar {SRC}/{KEY} {DEST} --progress',shell=True)