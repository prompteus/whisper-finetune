mkdir -p data

# source: https://github.com/mdeff/fma
# 8,000 tracks of 30s, 8 balanced genres
wget -O data/fma_small.zip https://os.unil.cloud.switch.ch/fma/fma_small.zip 
unzip data/fma_small.zip -d data/


# source: https://github.com/karolpiczak/ESC-50
# 
wget -O data/esc_50.zip https://github.com/karoldvl/ESC-50/archive/master.zip
unzip data/esc_50.zip -d data/
mv data/ESC-50-master data/esc_50
