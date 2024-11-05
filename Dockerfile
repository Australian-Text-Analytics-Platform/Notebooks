FROM milysun/atap-dev-env:juxtorpus

COPY . /workspace/
RUN pip install -e git+https://github.com/Australian-Text-Analytics-Platform/atap-context-extractor.git#egg=atap-context-extractor
RUN pip install --force-reinstall panel==1.5.2
RUN pip install --upgrade pybind11 pandas pyarrow juxtorpus

