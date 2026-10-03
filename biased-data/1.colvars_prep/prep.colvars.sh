#!/bin/bash
# This is a simple script to prepare colvars to be
# used by rateBiased.py

path="/Projects/afakhar/Muller/US/out/"
ni=20
nc=5
rm -rf colvars.txt
for i in `seq 1 $ni`; do
for j in `seq 1 $nc`; do
if [[ -f $path/$i.$j.gz ]]; then
zcat $path/$i.$j.gz | awk -v win=$((i-1)) -v cop=$((j-1)) '!/^#/ && $1 > 1.0 {print int($1*1000000), win, cop, $2, $3, "1.0" }' >> colvars.txt # time, win_id, copy_id, colvars
echo "-1 -1 -1 1000 1000 0" >> colvars.txt
else
echo "Warning: $path/$i.$j.gz not found"
fi
done
done

rm -rf colvarsd.txt
awk -v density="../0.fe_npwham/density.txt" '
$1 == -1 {print; next}
{ 
getline line < density
split(line, d)
if ($2 != d[1] || $1 != d[2]) {print "Error: colvars/density mismatch" > "/dev/stderr"; exit 1}
print $1, $2, $3, $4, $5, d[3]
}' colvars.txt > colvarsd.txt

# We convert the above file to npy format to be uploaded online
python<<EOF
import numpy as np
data = np.loadtxt('colvarsd.txt')
np.save('colvarsd_full.npy', data)    # time, window, copy, colvars, weight
np.save('colvarsd.npy', data[:, 3:6]) # x, y, weight; for PHSM
EOF

rm -rf colvarsd.txt colvars.txt
