#!/bin/sh

# -------------------------------------------------------
# Constants to specify by the user:
DATA_DIR=/home/dari/Projects/pointrendering2/Code/scenes/church/
# -------------------------------------------------------

# The following constants are filled in automatically:
NAME=npbg
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
SRC_DIR=`echo $SCRIPTPATH/../..`
# WORK_DIR=/home/docker/src/npbg
WORK_DIR=/home/docker/src/
#WORK_DIR=/home/docker/src/asdf
#WORK_DIR=/home/dari/Projects/pointrendering2/Code/External/npbg/

USER=`whoami`
TAG=$USER/$NAME:latest

export NAME
export TAG
export SRC_DIR
export DATA_DIR
export WORK_DIR
