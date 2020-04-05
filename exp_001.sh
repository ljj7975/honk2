mkdir log_001

echo "cnn-trad-fpool3.json"
date +"%r"
python -m run.train --config config/exp/001/cnn-trad-fpool3.json &> log_001/cnn-trad-fpool3.out

echo "cnn-one-fpool3.json"
date +"%r"
python -m run.train --config config/exp/001/cnn-one-fpool3.json &> log_001/cnn-one-fpool3.out

echo "cnn-one-fstride4.json"
date +"%r"
python -m run.train --config config/exp/001/cnn-one-fstride4.json &> log_001/cnn-one-fstride4.out

echo "cnn-one-fstride8.json"
date +"%r"
python -m run.train --config config/exp/001/cnn-one-fstride8.json &> log_001/cnn-one-fstride8.out

echo "cnn-tstride2.json"
date +"%r"
python -m run.train --config config/exp/001/cnn-tstride2.json &> log_001/cnn-tstride2.out

echo "cnn-tstride4.json"
date +"%r"
python -m run.train --config config/exp/001/cnn-tstride4.json &> log_001/cnn-tstride4.out

echo "cnn-tstride8.json"
date +"%r"
python -m run.train --config config/exp/001/cnn-tstride8.json &> log_001/cnn-tstride8.out

echo "cnn-tpool2.json"
date +"%r"
python -m run.train --config config/exp/001/cnn-tpool2.json &> log_001/cnn-tpool2.out

echo "cnn-tpool3.json"
date +"%r"
python -m run.train --config config/exp/001/cnn-tpool3.json &> log_001/cnn-tpool3.out
