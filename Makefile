stencil: stencil.c
	mpicc -std=c99 -O3 -Wall $^ -o $@

