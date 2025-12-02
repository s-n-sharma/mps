#!/usr/bin/env bash
xcrun -sdk macosx metal -c fma_sparse.metal -o fma_sparse.air && \
xcrun -sdk macosx metallib fma_sparse.air -o default.metallib && \
swiftc main.swift -o benchmark && \
./benchmark
