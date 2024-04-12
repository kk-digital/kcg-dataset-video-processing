# kcg-dataset-video-dataset-processing

# environment setup

install ffmpeg and other libraries with conda
```bash
conda create -p /opt/envs/video -c conda-forge mamba ffmpeg-python
conda activate /opt/envs/video
conda install Pillow numpy pandas matplotlib py-opencv -c conda-forge
conda install ipython ipykernel ipywidgets nbformat tqdm -c conda-forge
apt-get install libopengl0 libegl1
```
