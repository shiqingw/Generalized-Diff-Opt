FROM continuumio/miniconda3

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive
ENV PATH="/usr/local/bin:${PATH}"

# Run package updates, install packages, and then clean up to reduce layer size
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential cmake g++ git wget libatomic1 gfortran perl m4 cmake pkg-config \
    libopenblas-dev libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Update Python in the base environment to 3.8
RUN conda install python==3.8.16 \
    && conda clean -afy

# Download and build Julia for ARM (this will take quite a while)
RUN git clone https://github.com/JuliaLang/julia.git \
    && cd julia \
    && git checkout v1.9.3 \
    && make \
    && rm -rf /julia/deps \
    && rm -rf /julia/test

# Link Julia executable to a directory in PATH
RUN ln -s /julia/usr/bin/julia /usr/local/bin/julia
RUN julia -e 'using Pkg; Pkg.add("PyCall")'

# # Install Python packages via pip and then remove cache to reduce layer size
RUN pip install numpy==1.24.4 \
    scipy==1.10.1 \
    matplotlib==3.7.2 \
    cvxpy==1.4.1 \
    cvxpylayers==0.1.6 \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    open3d==0.17.0 \
    opencv-python==4.8.1.78 \
    proxsuite==0.5.0 \
    pybullet==3.2.5 \
    gymnasium==0.29.1 \
    apriltag==0.0.16 \
    pin==2.6.20 \
    julia \
    && rm -rf ~/.cache/pip

RUN python -c "import julia; julia.install()"

# Install Julia packages and then remove cache to reduce layer size
RUN julia -e 'using Pkg; Pkg.add(["ReverseDiff", "ForwardDiff", "ImplicitDifferentiation"]);'
RUN julia -e 'using Pkg; Pkg.add(["StaticArrays"]);'
RUN julia -e 'using Pkg; Pkg.add(["SCS", "ECOS", "Ipopt"]);'
RUN julia -e 'using Pkg; Pkg.add(["Convex", "JuMP", "DiffOpt"]);'
RUN julia -e 'using Pkg; Pkg.add(["LinearAlgebra", "Random", "Zygote"]);'

# Spin the container
CMD ["tail", "-f", "/dev/null"]