#/bin/bash

BR1=$1
BR2=$2
ERR=0

prepare_version (){
  BR=$1
  mkdir -p tmp
  git clone -b ${BR} ../cuckoo tmp/${BR}
  cp Makefile tmp/${BR}
  (cd tmp/${BR} ; make)
}

compare () {
  CMD="$1 -h $2"
  RES1=`tmp/${BR1}/bin/${CMD} | grep Sol | wc -l`
  RES2=`tmp/${BR2}/bin/${CMD} | grep Sol | wc -l`
  if [ $RES1 -ne $RES2 ]; then
    ERR=$(( $ERR + 1 ))
    echo "ERROR: lean15 -n $N differs!!"
  else
    echo "OK: ${CMD}  (result: $RES1)"
  fi
}

if [ $# -lt 2 ]; then
  echo "Usage: $0 <branch1> <branch2>"
fi

echo "Comparing ${BR1} with ${BR2}"

prepare_version ${BR1}
prepare_version ${BR2}

for H in "abcde_6" "abcde_97" "abcde_100"; do
  compare "lean15-generic     " $H
  compare "mean15-generic     " $H
#  compare "lean29-generic -t 4" $H
#  compare "lean29-avx2 -t 4" $H
  compare "mean29-generic -t 4" $H
  compare "mean29-avx2    -t 4" $H
done

if [ $ERR -lt 1 ]; then
  echo "TEST: PASSED"
else
  echo "TEST: FAILED ${ERR} errors"
fi

rm -rf tmp
