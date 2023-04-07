echo "Check Src"
ls /home/data/1945 | wc -l
cp -ar /home/data/1945 /home/dd
echo "Copy Done"
ls /home/dd | wc -lservice sshd start