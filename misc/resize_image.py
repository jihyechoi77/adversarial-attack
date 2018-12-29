# Resize all images in a directory to the designated size.
#
# Input is the directory where the original files are located
#
# If the script is in /images/ and the files are in /images/2012-1-1-pics
# call with: python resize.py 2012-1-1-pics

import Image
import os
import sys

data_dir = sys.argv[1]
save_dir = sys.argv[2]
# new_dim = [int(sys.argv[2])]  # for example, 224
new_dim = 160
cropped = 45

# for celeba
"""
for file_name in os.listdir(data_dir):
  print("Processing %s" % file_name)
  image = Image.open(os.path.join(data_dir, file_name))
  w, h = image.size

  # crop
  image = image.crop((cropped, cropped, w-cropped, h-cropped))

  # resize
  output = image.resize((new_dim, new_dim), Image.ANTIALIAS)

  output_file_name = os.path.join(save_dir, file_name)
  output.save(output_file_name, "JPEG", quality = 95)

print("Completed")
"""

# for lfw / vgg
for ith_sbj_name in os.listdir(data_dir):
  print("Processing %s" % ith_sbj_name)
  data_dir_full = os.path.join(data_dir, ith_sbj_name)

  for file_name in os.listdir(data_dir_full):
    image_path = os.path.join(data_dir_full, file_name)
    image = Image.open(image_path)
    w, h = image.size

    # crop
    image = image.crop((cropped, cropped, w - cropped, h - cropped))

    # resize
    output = image.resize((new_dim, new_dim), Image.ANTIALIAS)

    # output_file_name = os.path.join(os.path.join(save_dir, ith_sbj_name), file_name)
    output_file_name = os.path.join(os.path.join(save_dir, ith_sbj_name), file_name)
    output.save(output_file_name, "JPEG", quality=95)
