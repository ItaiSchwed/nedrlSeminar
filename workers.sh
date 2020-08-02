for i in {0..30}
do
nohup rq worker -c settings > workers/$i.log &
done
