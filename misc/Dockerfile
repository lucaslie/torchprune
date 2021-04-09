# Base Image is Ubuntu 18.04 with CUDA 11.0
FROM nvidia/cuda:11.0-devel-ubuntu18.04

# Make bash the default shell
SHELL ["/bin/bash", "-c"]

# Install a bunch of useful packages and miniconda
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y g++ wget libxrender1 vim git && \
    wget --quiet -O ~/miniconda.sh \
    https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# add conda to environment path variable
ENV PATH /opt/conda/bin:$PATH

# create a custom bashrc that is run even for non-interactive shells
# also explicitly source from all standard bash config files
RUN touch /etc/bashrc_custom && \
    conda init bash && \
    awk '/# >>> conda initialize >>>/,/# <<< conda initialize <<</' ~/.bashrc \
    >> /etc/bashrc_custom && \
    echo -e "\numask 002\n" >> /etc/bashrc_custom && \
    echo -e "\nsource /etc/bashrc_custom\n" >> /etc/bash.bashrc && \
    echo -e "\nsource /etc/bashrc_custom\n" >> /etc/profile && \
    echo -e "\nsource /etc/bashrc_custom\n" >> /root/.bashrc

# tell bash to source the custom bashrc when a shell is started
ENV BASH_ENV /etc/bashrc_custom

# Copy source files and requirements file
COPY ./src /src
COPY ./misc /misc
COPY ./paper /paper
COPY ./misc/requirements.txt /misc/requirements.txt

# Install the required packages
# also include headless opencv
RUN conda install -y pip python=3.8 && \
    pip install -r misc/requirements.txt opencv-python-headless

# Workdir is the root directory
WORKDIR /

# Declare data volume
VOLUME /data

# Declare local volume
VOLUME /local

# default entrypoint is an non-interactive, non-login shell
ENTRYPOINT ["/bin/bash", "-c"]

# run some sample commands as default command
CMD ["echo -e 'Hello World'; nvidia-smi"]
