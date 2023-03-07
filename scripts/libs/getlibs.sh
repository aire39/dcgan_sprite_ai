#!/bin/bash

echo "Get Pytorch libs!"

# download pytorch library zip files

PYTORCH_1_13_0_RELEASE=('1tXFqNXUfCA8sz-aQjOeRVmlCLeoq-fjN' 'pytorch_1.13.0_release.zip')
PYTORCH_1_13_0_DEBUG=('1WcDzi6uLbt4Icf_7r9lgag7ZtuqhHWSr' 'pytorch_1.13.0_debug.zip')

sh ../utility/googledrive_file_download.sh ${PYTORCH_1_13_0_RELEASE[0]} ${PYTORCH_1_13_0_RELEASE[1]} 
sh ../utility/googledrive_file_download.sh ${PYTORCH_1_13_0_DEBUG[0]} ${PYTORCH_1_13_0_DEBUG[1]}

# extraction of the pytorch library files. The check for 7z should only have to happen once

7z x pytorch_1.13.0_release.zip -o../../libs/pytorch_1_13_0_release

if [ $? -eq 0 ]; then
	echo "extracting successful!"
else
	echo "7zip missing to extract archived file! Will download command..."
	pacman -S 7zip
	echo "complete... Will contunie with extracting files!"
	7z x pytorch_1.13.0_release.zip -o../../libs/pytorch_1_13_0_release
fi

7z x pytorch_1.13.0_debug.zip -o../../libs/pytorch_1_13_0_debug

# cleanup files

rm pytorch_1.13.0_release.zip
rm pytorch_1.13.0_debug.zip
