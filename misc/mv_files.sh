#!/bin/bash

#for ((i=1; i<=5; i++)); do
#    cp -r ./1${i}*.jpg train/
#done

#for ((i=0; i<=1; i++)); do
#    cp -r ./16${i}*.jpg train/
#done
#
#for ((i=0; i<=6; i++)); do
#    cp -r ./162${i}*.jpg train/
#done
#
#for ((i=0; i<=6; i++)); do
#    cp -r ./1627${i}*.jpg train/
#done
#cp ./162770.jpg train/
#
#
#
#cp ./182638.jpg test/
#cp ./182639.jpg test/
#for ((i=4; i<=9; i++)); do
#    cp -r ./1826${i}*.jpg test/
#done
#for ((i=7; i<=9; i++)); do
#    cp -r ./182${i}*.jpg test/
#done
#for ((i=3; i<=9; i++)); do
#    cp -r ./18${i}*.jpg test/
#done
#cp -r ./19*.jpg test/
#


for ((i=1; i<=9; i++)) do
    cp -r ./16277${i}*.jpg validation/
done
for ((i=8; i<=9; i++)) do
    cp -r ./1627${i}*.jpg validation/
done
for ((i=8; i<=9; i++)) do
    cp -r ./162${i}*.jpg validation/
done
for ((i=3; i<=9; i++)) do
    cp -r ./16${i}*.jpg validation/
done
cp -r ./17*.jpg validation/
cp -r ./180*.jpg validation/
cp -r ./181*.jpg validation/
for ((i=0; i<=5; i++)) do
    cp -r ./182${i}*.jpg validation/
done
for ((i=0; i<=2; i++)) do
    cp -r ./1826${i}*.jpg validation/
done
for ((i=0; i<=7; i++)) do
    cp -r ./18263${i}*.jpg validation/
done
