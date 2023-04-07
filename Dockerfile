FROM continuumio/anaconda3:latest
# Install necessary packages
RUN apt update && apt -y upgrade && apt -y install libcurl4 g++ make
# Create a conda environment and install conda packages
RUN conda create -n torch python=3.9
# Set the default shell to start with the torch conda environment
SHELL [ "conda", "run", "-n", "torch", "/bin/bash", "-c"]
# Install pytorch with cuda 11.8, jupyter, transformers, datasets, xformers, and triton
RUN conda install pytorch torchvision torchaudio torchtext pytorch-cuda=11.8 -c pytorch -c nvidia && \
    conda install -c anaconda jupyter && \
    pip install transformers datasets xformers triton
# Add the torch conda environment to jupyter lab
RUN ipython kernel install --user --name=torch
# Expose port 8888 that jupyter will use
EXPOSE 8888
# Set the default command to run when starting the container
CMD ["conda", "run", "--no-capture-output", "-n", "torch", "python", "-m", "jupyter", "lab", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]