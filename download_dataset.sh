wget -O vctk.zip "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip?sequence=2&isAllowed=y" --no-check-certificate
wget -O wham.zip https://storage.googleapis.com/whisper-public/wham_noise.zip
git clone https://github.com/microsoft/MS-SNSD.git
unzip vctk.zip
unzip wham_noise.zip
mkdir bf_cache
mkdir bf_cache_test
mkdir temp
mkdir trained