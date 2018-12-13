#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define THREADS 16

/*
gcc -fopenmp -std=c99 -O3 -Wall stencil.c -o stencil
*/

void stencil(const int z, const int nx, const int ny, float *  image, float *  tmp_image, const int left, const int right);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
double wtime(void);

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initiliase problem dimensions from command line arguments
  int numx = atoi(argv[1]);
  int numy = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  float *image = malloc(sizeof(double)*numx*numy);
  float *tmp_image = malloc(sizeof(double)*numx*numy);

  // Set the input image
  init_image(numx, numy, image, tmp_image);

  double tic, toc;

  omp_set_num_threads(THREADS);

  int rank, size, nx, ny, z, t, left, right;

#pragma omp parallel default(shared) private(rank,size,nx,ny,z,t,left,right)
  {
    left = 0; right = 0;

    rank = omp_get_thread_num();
    size = omp_get_num_threads();

    if (rank == 0)
      printf("openMP: %d threads, img %d\n", size, numx);

    // determine portion width + height
    nx = numx / size;
    ny = numy;

    // index of top left cell of portion
    z = (rank * nx) * ny;

    // account for input sizes not divisible by cohort size
    if (numx % size != 0) {
      if (rank < numx % size)
        nx++;
      z += ny*((rank > numx % size) ? numx % size : rank);
    }

    // add 1 or two to nx depending on how many halo cols
    // also shift starting index back 1 col if not first
    if (rank == 0) {
      left = 1;
      nx += 1;
    } else if (rank == size - 1) {
      right = 1;
      z -= ny;
      nx += 1;
    } else {
      z -= ny;
      nx += 2;
    }

    #pragma omp barrier

    if (rank == 0)
      tic = wtime();

    // Call the stencil kernel
    for (t = 0; t < niters; ++t) {
      stencil(z, nx, ny, image, tmp_image, left, right);
      #pragma omp barrier
      stencil(z, nx, ny, tmp_image, image, left, right);
      #pragma omp barrier
    }

    if (rank == 0)
      toc = wtime();
  }

  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc-tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, numx, numy, image);
  free(image);
  free(tmp_image);
}

/**
 * z    top left corner
 * nx   num cols
 * ny   num rows
 */
void stencil(const int z, const int nx, const int ny, float * restrict image, float * restrict tmp_image, const int left, const int right) {

  if (left == 1) {
    // top left corner
    tmp_image[z] = image[z] * 0.6f + image[z+ny] * 0.1f + image[z+1] * 0.1f;

    // 'left' vertical edge cells
    for (int j = 1; j != ny-1; ++j) {  // VECTORIZING
      // i = 0
      tmp_image[z+j] =  image[z+j] * 0.6f + image[z+j+ny] * 0.1f + image[z+j-1] * 0.1f + image[z+j+1] * 0.1f;
    }

    // bottom left corner
    tmp_image[z+ny-1] = image[z+ny-1] * 0.6f + image[z+2*ny-1] * 0.1f + image[z+ny-2] * 0.1f;
  }

  // non-edge cells
  for (int i = 1; i != nx-1; ++i) {
    // j = 0
    tmp_image[z+i*ny] = image[z+i*ny] * 0.6f + image[z+(i-1)*ny] * 0.1f + image[z+(i+1)*ny] * 0.1f + image[z+1+i*ny] * 0.1f;

    for (int j = 1; j != ny-1; ++j) {  // VECTORIZING
      tmp_image[z+j+i*ny] = image[z+j+i*ny] * 0.6f + image[z+j+i*ny-ny] * 0.1f + image[z+j+i*ny+ny] * 0.1f + image[z+j+i*ny-1] * 0.1f + image[z+j+i*ny+1] * 0.1f;
    }

    // j = (ny-1)
    tmp_image[z+ny-1+i*ny] = image[z+ny-1+i*ny] * 0.6f + image[z+ny-1+(i-1)*ny] * 0.1f + image[z+ny-1+(i+1)*ny] * 0.1f + image[z+ny-2+i*ny] * 0.1f;
  }

  if (right == 1) {
    // i = (nx-1), j = 0
    tmp_image[z+(nx-1)*ny] = image[z+(nx-1)*ny] * 0.6f + image[z+(nx-2)*ny] * 0.1f + image[z+1+(nx-1)*ny] * 0.1f;

    // 'right' vertical edge cells
    for (int j = 1; j != ny-1; ++j) {  // VECTORIZING
      // i = (nx-1)
      tmp_image[z+j+(nx-1)*ny] = image[z+j+(nx-1)*ny] * 0.6f + image[z+j+(nx-2)*ny] * 0.1f + image[z+j-1+(nx-1)*ny] * 0.1f + image[z+j+1+(nx-1)*ny] * 0.1f;
    }

    // i = (nx-1), j = ny-1
    tmp_image[z-1+nx*ny] =  image[z-1+nx*ny] * 0.6f + image[z-1+(nx-1)*ny] * 0.1f + image[z-2+nx*ny] * 0.1f;
  }
}

// Create the input image
void init_image(const int nx, const int ny, float *  image, float *  tmp_image) {
  // Zero everything
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      image[j+i*ny] = 0.0;
      tmp_image[j+i*ny] = 0.0;
    }
  }

  // Checkerboard
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      for (int jj = j*ny/8; jj < (j+1)*ny/8; ++jj) {
        for (int ii = i*nx/8; ii < (i+1)*nx/8; ++ii) {
          if ((i+j)%2)
          image[jj+ii*ny] = 100.0;
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char * file_name, const int nx, const int ny, float *image) {

  // Open output file
  FILE *fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      if (image[j+i*ny] > maximum)
        maximum = image[j+i*ny];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      fputc((char)(255.0*image[j+i*ny]/maximum), fp);
    }
  }

  // Close the file
  fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec*1e-6;
}