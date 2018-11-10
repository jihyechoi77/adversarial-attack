# Split and save files in a directory to multiple directories.
# Open the directory in terminal and run this.


i=0;
n=10000; # number of images in a directory 

for f in *; 
do 
    d=../img_align_celeba_dim224_dir$(printf %06d $((i/n+1))); 
    mkdir -p $d; 
    mv "$f" $d; 
    let i++; 
done
