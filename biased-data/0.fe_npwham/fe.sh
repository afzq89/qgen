#!/bin/bash
# This is a simple script to calcualte the biased potentials;
# th free energy, and density weights from npwham.

K=100
temp=300
rate=1
ni=20
nc=5 # just for demo

current_dir=$PWD
cd /Projects/afakhar/Muller/US/prm
C="`grep center * | sort -n | awk '{printf "%s %s ",$2,$3}'`"
cd $current_dir
if (( 1 ))
then

rm -rf pot.txt
    for i in `seq 1 $ni`;do
    for j in `seq 1 $nc`;do
        zcat /Projects/afakhar/Muller/US/out/$i.$j.gz | \
        # npwham likes to start from zero!
        awk -v W=$((i-1)) -v co=$((j-1)) -v K=$K -v rate=$rate -v C="$C" 'BEGIN{split(C,c)}
            ($1 >  1.0 && $1 !~ /#/&&NR%rate==0) {
                printf "%d %d %d", W, co, int($1*1000000) # Window, copy, time
                x=$2
                y=$3
                for (I=0;I<=19;I++) {
                    xc=c[2*I+1]
                    yc=c[2*I+2]
                    U=0.5*K*((x-xc)*(x-xc)+(y-yc)*(y-yc))
                    printf " %f",U
                }
                printf "\n";
            }' >> pot.txt
    done
    done
fi

l=`wc pot.txt | awk '{print $1}'`
../../../../npwham.0/npwham -w $ni -l $l -t $temp -i 1000 -b $nc -B 1 < pot.txt > density.txt 2> density.err
tail -$ni density.err | awk '{print $3,$4,$5}' > fe.txt
