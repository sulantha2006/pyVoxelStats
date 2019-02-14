FROM ubuntu:16.04
MAINTAINER Sulantha Mathotaarachchi <sulantha.s@gmail.com>

# System packages
RUN apt-get update && apt-get install -y curl libc6 libstdc++6 imagemagick perl bzip2 bash

# MINC
RUN curl -LO http://packages.bic.mni.mcgill.ca/minc-toolkit/Debian/minc-toolkit-1.9.16-20180117-Ubuntu_16.04-x86_64.deb
RUN dpkg -i minc-toolkit-1.9.16-20180117-Ubuntu_16.04-x86_64.deb
RUN apt-get install -f
RUN rm minc-toolkit-1.9.16-20180117-Ubuntu_16.04-x86_64.deb
# PYTHON
RUN apt-get install -y python3 python3-pip software-properties-common apt-transport-https
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'
RUN apt-get update
RUN apt-get install -y r-base
RUN pip3 install nibabel numpy pandas pyminc statsmodels rpy2 ipyparallel numexpr scipy pyvoxelstats

# ENV
ENV MINC_TOOLKIT="/opt/minc/1.9.16"
ENV MINC_TOOLKIT_VERSION="1.9.16-20180117"
ENV PERL5LIB="${MINC_TOOLKIT}/perl:${MINC_TOOLKIT}/pipeline:${PERL5LIB}"
ENV LD_LIBRARY_PATH="${MINC_TOOLKIT}/lib:${MINC_TOOLKIT}/lib/InsightToolkit:${LD_LIBRARY_PATH}"
ENV MINC_FORCE_V2=1
ENV MINC_COMPRESS=4
ENV VOLUME_CACHE_THRESHOLD=-1
ENV MANPATH="${MINC_TOOLKIT}/man:${MANPATH}"
ENV PATH="${MINC_TOOLKIT}/bin:${MINC_TOOLKIT}/pipeline:${PATH}"

ENTRYPOINT ["/bin/bash"]
