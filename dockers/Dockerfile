FROM python:3.8-bullseye 

RUN curl -L -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install numpy pyyaml scipy ipython mkl && \
     /opt/conda/bin/conda install -c soumith magma-cuda90 && \
     /opt/conda/bin/conda clean -ya 

ENV PATH /opt/conda/bin:$PATH 
# This must be done before pip so that requirements.txt is available
WORKDIR /opt/pytorch

RUN conda install pytorch==1.7.0 cpuonly -c pytorch

RUN pip install --upgrade pip \
    pip install setuptools  \
    pip install numpy \
    pip install sklearn \
    pip install pandas \
    pip install pyarrow \
    pip install openpyxl 

CMD ["python", "-u"]
