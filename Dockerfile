FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

WORKDIR /app

# 参照用にCOPY
COPY requirements_docker.txt /tmp/requirements_docker.txt

# Pip のアップグレードはホスト環境によってビルド中に不安定になるため省略
# ここでは現状の pip バージョンを確認するだけにして、必要ならコンテナ内で個別に更新してください。
RUN python -m pip --version

# --- 1. Base Science Stack (use conda to avoid pip resolver/build issues) ---
# Install heavy numerical/data packages via conda (conda-forge) which provides
# prebuilt binaries and avoids pip's complex resolver during image build.
ARG INSTALL_SCIENCE_PACKAGES=false
# By default skip heavy scientific package installation during image build.
# If you have sufficient build resources and want them baked in, build with:
#   docker build --build-arg INSTALL_SCIENCE_PACKAGES=true .
RUN if [ "$INSTALL_SCIENCE_PACKAGES" = "true" ] ; then \
            conda install -y -c conda-forge \
                numpy=1.26.4 \
                scipy=1.13.1 \
                pandas=1.5.3 \
                scikit-learn=1.6.1 \
                matplotlib=3.9.4 \
            && conda clean -afy ; \
        else \
            echo "Skipping heavy science packages (set INSTALL_SCIENCE_PACKAGES=true to enable)" ; \
        fi

# --- 2. EEG & DL Domain ---
RUN python -m pip install --no-cache-dir \
    mne==1.6.1 braindecode==0.8.1 moabb==1.0.0 torchinfo==1.8.0

# --- 3. Utilities ---
RUN python -m pip install --no-cache-dir \
    pyyaml==6.0.2 tqdm==4.67.1

# --- 4. Dev Tools & MLOps ---
# Omitted heavy dev tools (jupyterlab, ipywidgets, mlflow) to avoid build-time OOM/segfault.
# Install them later in the running container if needed.
# RUN python -m pip install --no-cache-dir \
#     "jupyterlab>=4.0.0" ipywidgets mlflow

# --- 5. LLM client libraries (optional, may require more build resources) ---
ARG INSTALL_LLM_PACKAGES=false
RUN if [ "$INSTALL_LLM_PACKAGES" = "true" ] ; then \
            python -m pip install --no-cache-dir \
                langchain-ollama langchain-core ollama \
            && python -m pip install --no-cache-dir langgraph ; \
        else \
            echo "Skipping LLM client packages (set INSTALL_LLM_PACKAGES=true to enable)" ; \
        fi

CMD ["tail", "-f", "/dev/null"]