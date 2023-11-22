FROM debian:bullseye

RUN apt-get update && \ 
apt-get install -y g++ wget bzip2 libarchive-tools && \
apt-get clean && \ 
rm -rf /var/lib/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda/ && \
    rm miniconda.sh

WORKDIR /home

ENV PATH=/opt/conda/bin:$PATH

COPY torch_env.yml .

RUN conda install -y -c conda-forge mamba && \
    mamba env create -f torch_env.yml && \ 
    conda run -n pytorch pip install torch_geometric && \ 
    conda run -n pytorch pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html && \
    rm *env.yml && \ 
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    echo "conda activate pytorch" >> /etc/bash.bashrc

CMD ["/bin/bash","-l"]