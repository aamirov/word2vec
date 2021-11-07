make
if [ ! -e text8 ]; then
  wget http://mattmahoney.net/dc/text8.zip -O text8.gz
  gzip -d text8.gz -f
fi
#time ./word2vec -train text8 -output vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
time ./word2vec -train text8 -output vectors.bin -cbow 0 -size 128 -window 10 -negative 25 -hs 0 -sample 1e-4 -threads 2 -binary 0 -iter 1
.
./distance vectors.bin


#	call("/mnt/word2vec/word2vec/w2v/Word2Vec/word2vec/word2vec -train {} -output {} -cbow 0 -size {} -window 10 \
#				 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15".format(rwalks_path, outputEmbs, args.emb_dims), shell=True)
		