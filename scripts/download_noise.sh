mkdir -p data

# source: https://github.com/mdeff/fma
# 8,000 tracks of 30s, 8 balanced genres
wget -O data/FMA-small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip 
unzip data/FMA-small.zip -d data/


# source: https://github.com/karolpiczak/ESC-50
wget -O data/esc_50.zip https://github.com/karoldvl/ESC-50/archive/master.zip
unzip data/esc_50.zip -d data/
mv data/ESC-50-master data/ESC-50

# or download several parts of a larger ESC dataset
# this can be done manually from:
#    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT
# place the files into data folder and run:
find data/ -name ESC-US-* -print0 | parallel -0 tar -xvzf {} -C data/
