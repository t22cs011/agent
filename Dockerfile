FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

WORKDIR /app

# 参照用にCOPY
COPY requirements_docker.txt /tmp/requirements_docker.txt

# Pipのアップグレード
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# --- 1. Base Science Stack (分割インストールでSegfault回避) ---
# NumPyは最もクラッシュしやすいため単独かつ最初にインストール
RUN python -m pip install --no-cache-dir numpy==1.26.4

# SciPyもコンパイル済みバイナリが重いため単独
RUN python -m pip install --no-cache-dir scipy==1.13.1

# Pandas
RUN python -m pip install --no-cache-dir pandas==1.5.3

# その他の分析系ライブラリ
RUN python -m pip install --no-cache-dir scikit-learn==1.6.1 matplotlib==3.9.4

# --- 2. EEG & DL Domain ---
RUN python -m pip install --no-cache-dir \
    mne==1.6.1 braindecode==0.8.1 moabb==1.0.0 torchinfo==1.8.0

# --- 3. Utilities ---
RUN python -m pip install --no-cache-dir \
    pyyaml==6.0.2 tqdm==4.67.1

# --- 4. Dev Tools & MLOps ---
RUN python -m pip install --no-cache-dir \
    "jupyterlab>=4.0.0" ipywidgets mlflow

CMD ["tail", "-f", "/dev/null"]