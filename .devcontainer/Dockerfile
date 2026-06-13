FROM ocaml/opam:debian-12-ocaml-5.5

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    m4 \
    pkg-config \
    libgmp-dev \
    libcurl4-gnutls-dev \
    ca-certificates \
    curl \
    less \
    && rm -rf /var/lib/apt/lists/*
USER opam

RUN opam update && opam install -y \
    dune \
    utop \
    ocamlformat \
    ocaml-lsp-server \
    odoc \
    ppx_expect

RUN echo 'eval $(opam env)' >> ~/.bashrc
