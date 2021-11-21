FROM continuumio/anaconda3
ADD environment.yml environment.yml
RUN conda env create -f environment.yml
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH
RUN conda init
COPY . /
EXPOSE 8061
CMD ["conda", "run", "--no-capture-output", "-n", "dao", "python","dao_server.py" ]
