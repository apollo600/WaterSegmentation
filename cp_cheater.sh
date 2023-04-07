echo "Check Src"
ls /home/data/1945 | wc -l
cp -ar /home/data/1945 /project/train/src_repo/1945
echo "Copy Done"
ls /project/train/src_repo/1945 | wc -l