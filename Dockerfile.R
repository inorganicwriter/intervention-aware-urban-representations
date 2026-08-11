FROM rocker/r-base:4.6.1

# System dependencies for compiling R packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev libcurl4-openssl-dev libssl-dev \
    libgdal-dev libgeos-dev libproj-dev \
    libharfbuzz-dev libfribidi-dev libfreetype6-dev \
    libpng-dev libtiff5-dev libjpeg-dev \
    liblapack-dev libopenblas-dev libarmadillo-dev \
    gcc g++ gfortran make \
    && rm -rf /var/lib/apt/lists/*

# R package library
RUN mkdir -p /home/rstudio/.r-lib
ENV R_LIBS_USER=/home/rstudio/.r-lib

# Install core CRAN packages
RUN R -e 'install.packages(c( \
    "Rcpp", "RcppArmadillo", "RcppEigen", \
    "data.table", "dplyr", "arrow", \
    "fixest", "lmtest", "sandwich", \
    "Matching", "PanelMatch", \
    "future", "doParallel", "foreach" \
  ), repos = "https://cloud.r-project.org")'

# gsynth and fect (CRAN)
RUN R -e 'install.packages(c("gsynth", "fect"), \
    repos = "https://cloud.r-project.org")'

# Verify
RUN R -e 'for(pkg in c("data.table","arrow","Matching","PanelMatch", \
    "gsynth","fect","future","fixest","dplyr")) { \
    cat(sprintf("%-20s %s\n", pkg, packageVersion(pkg))) }'

WORKDIR /home/rstudio/project

CMD ["R"]
