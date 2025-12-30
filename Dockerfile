FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

WORKDIR /app

COPY requirements_docker.txt /tmp/requirements_docker.txt

# 分割インストールでセグフォを回避
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 scipy==1.13.1 pandas==1.5.3 scikit-learn==1.6.1 matplotlib==3.9.4
RUN python -m pip install --no-cache-dir \
    mne==1.6.1 braindecode==0.8.1 moabb==1.0.0 torchinfo==1.8.0
RUN python -m pip install --no-cache-dir \
    pyyaml==6.0.2 tqdm==4.67.1
RUN python -m pip install --no-cache-dir \
    "jupyterlab>=4.0.0" ipywidgets

CMD ["tail", "-f", "/dev/null"]
