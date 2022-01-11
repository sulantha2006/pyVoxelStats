FROM ubuntu:20.04
MAINTAINER Sulantha Mathotaarachchi <sulantha.s@gmail.com>

ENV DEBIAN_FRONTEND=noninteractive
# System packages
RUN apt-get update && apt-get install -y curl libc6 libstdc++6 imagemagick perl bzip2 bash software-properties-common dirmngr wget

# MINC
RUN curl -LO https://packages.bic.mni.mcgill.ca/minc-toolkit/Debian/minc-toolkit-1.9.18-20200813-Ubuntu_20.04-x86_64.deb
RUN dpkg -i minc-toolkit-1.9.18-20200813-Ubuntu_20.04-x86_64.deb
RUN apt-get install -f
RUN rm minc-toolkit-1.9.18-20200813-Ubuntu_20.04-x86_64.deb
# PYTHON
RUN apt-get install -y python3 python3-pip software-properties-common apt-transport-https
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"
RUN apt-get update
RUN apt-get install -y r-base
RUN pip3 install -U --no-cache-dir nibabel numpy pandas pyminc statsmodels rpy2 ipyparallel numexpr scipy pyvoxelstats

# ENV
ENV MINC_TOOLKIT="/opt/minc/1.9.18"
ENV MINC_TOOLKIT_VERSION="1.9.18-20200813"
ENV PERL5LIB="${MINC_TOOLKIT}/perl:${MINC_TOOLKIT}/pipeline:${PERL5LIB}"
ENV LD_LIBRARY_PATH="${MINC_TOOLKIT}/lib:${MINC_TOOLKIT}/lib/InsightToolkit:${LD_LIBRARY_PATH}"
ENV MINC_FORCE_V2=1
ENV MINC_COMPRESS=4
ENV VOLUME_CACHE_THRESHOLD=-1
ENV MANPATH="${MINC_TOOLKIT}/man:${MANPATH}"
ENV PATH="${MINC_TOOLKIT}/bin:${MINC_TOOLKIT}/pipeline:${PATH}"

ENTRYPOINT ["/bin/bash"]
