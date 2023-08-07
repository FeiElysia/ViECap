SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/../evaluation

python cocoeval.py --result_file_path $1