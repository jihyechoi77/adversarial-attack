# Resize all images in a directory to the designated size.
#
# Inputs are 1) directory where the original files are located and 2) new dimension.
#
# If the script is in /images/ and the files are in /images/2012-1-1-pics
# call with: python resize.py 2012-1-1-pics

import Image
import os
import sys

directory = sys.argv[1]
# new_dim = [int(sys.argv[2])]  # for example, 224
new_dim = 160

for file_name in os.listdir(directory):
  print("Processing %s" % file_name)
  image = Image.open(os.path.join(directory, file_name))
  w, h = image.size

  # crop
  cropped = 15
  image = image.crop((cropped, cropped, w-cropped, h-cropped))

  # resize
  output = image.resize((new_dim, new_dim), Image.ANTIALIAS)

  output_file_name = os.path.join("%s-dim%d" %(directory, new_dim), file_name)
  output.save(output_file_name, "JPEG", quality = 95)

print("Completed")
