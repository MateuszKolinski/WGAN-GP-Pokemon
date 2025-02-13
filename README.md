# WGAN-GP Pokemon
# By Mateusz Kolinski MateuszPKolinski@gmail.com

This project bases on a WGAN-GP. It is a modified version of a stable CWGAN-GP for Polandball that I created here: https://github.com/MateuszKolinski/CWGAN-GP-Polandball

This version doesn't use conditional generation, doesn't pre- and post-transform images and uses different sets of generator/critic strides and kernels to better fit image dimensions. 

This project uses a very heavily augmented database of Pokemon sprites downloaded and modified in another project of mine: https://github.com/MateuszKolinski/PokemonSpriteDownloader

Tested for a handful of input images. Bigger sample size should work as well, but it is very time and resource consuming for me to use aforementioned database, hence I'm not showing any results below just yet.
