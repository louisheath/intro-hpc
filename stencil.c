#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// run with: mpirun -np 4 ./stencil 1024 1024 100

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

#define MASTER 0

void stencil(const int z, const int nx, const int ny, float *  image, float *  tmp_image);
void init_image(const int nx, const int ny, float *  image, float *  tmp_image);
void output_image(const char * file_name, const int nx, const int ny, float *image);
int mod(int x, int n);
double wtime(void);

/*
  image accessed by
    img[column * height + row]
*/

int main(int argc, char *argv[]) {

  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  // Initialise problem dimensions from command line arguments
  int numx = atoi(argv[1]);
  int numy = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // Allocate the image
  float *image = malloc(sizeof(float)*numx*numy);
  float *tmp_image = malloc(sizeof(float)*numx*numy);

  // Set the input image
  init_image(numx, numy, image, tmp_image);

  // Set up MPI
  int flag, rank, size, left, right, tag = 0;
  MPI_Status status;

  MPI_Init( &argc, &argv );
  MPI_Initialized(&flag);
  if ( flag != 1 )
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );

  right = rank + 1;
  left  = rank - 1;

  if (rank == MASTER)
    printf("%d nodes, %d %d %d\n", size, numx, numy, niters);
  // printf("rank %d, size %d, left %d, right %d\n", rank, size, left, right);

  // determine portion width + height
  int nx = numx / size;
  int ny = numy;

  // index of top left cell of portion
  int z = (rank * nx) * ny;

  // account for input sizes not divisible by cohort size
  if (numx % size != 0) {
    if (rank < numx % size)
      nx++;
    z += ny*((rank > numx % size) ? numx % size : rank);
  }
  
  // these indexes include halo columns
  int _nx = nx;
  int _ny = ny;
  int _z = z;

  // add 1 or two to nx depending on how many halo cols
  // also shift starting index back 1 col if not first
  if (rank == 0) {
    _nx += 1;
  } else if (rank == size - 1) {
    _nx += 1;
    _z -= ny;
  } else {
    _nx += 2;
    _z -= ny;
  }

  float haloL[ny]; // send buffers
  float haloR[ny];
  float haloN[ny]; // recv buffer

  // Start timing my code
  double tic = wtime();

  for (int t = 0; t < niters; ++t) {
    // change tmp_image
    stencil(_z, _nx, _ny, image, tmp_image);

    // update tmp_image halos
    if (rank != 0) {
      // prepare left halo
      for (int y = 0; y < ny; y++) {
        haloL[y] = tmp_image[z+y];
      }
    }
    if (rank != size - 1) {
      // prepare right halo
      for (int y = 0; y < ny; y++) {
        haloR[y] = tmp_image[z+(nx-1)*ny +y];
      }
    }

    // send halo cells
    if (rank == 0) {
      // receive from right
      MPI_Recv(haloN, ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);

      // put data in right halo
      for (int y = 0; y < ny; y++) {
        tmp_image[_z + (_nx-1)*ny + y] = haloN[y];
      }

      // send right
      MPI_Ssend(haloR, ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD);

    } else if (rank == size - 1) {
      // send left
      MPI_Ssend(haloL, ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD);

      // receive from left
      MPI_Recv(haloN, ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);
      // put data in left halo
      for (int y = 0; y < ny; y++) {
        tmp_image[_z+y] = haloN[y];
      }

    } else {
      // send left and receive from right
      MPI_Sendrecv(haloL, ny, MPI_FLOAT, left, tag,
	      haloN, ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);
      // put data in right halo
      for (int y = 0; y < ny; y++) {
        tmp_image[_z + (_nx-1)*ny + y] = haloN[y];
      }

      // send right and receive from left
      MPI_Sendrecv(haloR, ny, MPI_FLOAT, right, tag,
	      haloN, ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);
      // put data in left halo
      for (int y = 0; y < ny; y++) {
        tmp_image[_z+y] = haloN[y];
      }
    }

    // change image
    stencil(_z, _nx, _ny, tmp_image, image);

    // update image halos
    if (rank != 0) {
      // prepare left halo
      for (int y = 0; y < ny; y++) {
        haloL[y] = image[z+y];
      }
    }
    if (rank != size - 1) {
      // prepare right halo
      for (int y = 0; y < ny; y++) {
        // haloR[y] = image[((rank+1)*nx-1)*ny+y];
        haloR[y] = image[z+(nx-1)*ny +y];
      }
    }

    // send halo cells
    if (rank == 0) {
      // receive from right
      MPI_Recv(haloN, ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);

      // put data in right halo
      for (int y = 0; y < ny; y++) {
        image[_z + (_nx-1)*ny + y] = haloN[y];
      }

      // send right
      MPI_Ssend(haloR, ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD);
    } else if (rank == size - 1) {
      // send left
      MPI_Ssend(haloL, ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD);

      // receive from left
      MPI_Recv(haloN, ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);
      // put data in left halo
      for (int y = 0; y < ny; y++) {
        image[_z+y] = haloN[y];
      }

    } else {
      // send left and receive from right
      MPI_Sendrecv(haloL, ny, MPI_FLOAT, left, tag,
	      haloN, ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);
      // put data in right halo
      for (int y = 0; y < ny; y++) {
        image[_z + (_nx-1)*ny + y] = haloN[y];
      }

      // send right and receive from left
      MPI_Sendrecv(haloR, ny, MPI_FLOAT, right, tag,
	      haloN, ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);
      // put data in left halo
      for (int y = 0; y < ny; y++) {
        image[_z+y] = haloN[y];
      }
    }
  }

  // Stop timing my code
  double toc = wtime();

  float buffer[((numx/size)+1)*ny];

  // combine at master
  if (rank == MASTER) {
    for (int src = 1; src < size; src++) {

      
      int nx_s = numx / size;
      int z_s = src * nx_s * numy;
      
      if (numx % size != 0) {
        if (src < numx % size)
          nx_s++;
        z_s += numy*((src > numx % size) ? numx % size : src);
      }

      MPI_Recv(buffer, nx_s*ny, MPI_FLOAT, src, tag, MPI_COMM_WORLD, &status);
      for (int i = 0; i < nx_s*ny; i++)
        image[z_s+i] = buffer[i];
    }
  } else {
    for (int i = 0; i < nx*ny; i++)
      buffer[i] = image[z+i];
    MPI_Ssend(buffer, nx*ny, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
  }

  if (rank == MASTER) {
    // Output
    printf("------------------------------------\n");
    printf(" runtime: %lf s\n", toc-tic);
    printf("------------------------------------\n");

    output_image(OUTPUT_FILE, numx, numy, image);
  }
  
  // clean up
  free(image);
  free(tmp_image);
  MPI_Finalize();
}

/**
 * z    top left corner
 * nx   num cols
 * ny   num rows
 */
void stencil(const int z, const int nx, const int ny, float * restrict image, float * restrict tmp_image) {

  // top left corner
  tmp_image[z] = image[z] * 0.6f + image[z+ny] * 0.1f + image[z+1] * 0.1f;

  // 'left' vertical edge cells
  for (int j = 1; j != ny-1; ++j) {  // VECTORIZING
    // i = 0
    tmp_image[z+j] =  image[z+j] * 0.6f + image[z+j+ny] * 0.1f + image[z+j-1] * 0.1f + image[z+j+1] * 0.1f;
  }

  // bottom left corner
  tmp_image[z+ny-1] = image[z+ny-1] * 0.6f + image[z+2*ny-1] * 0.1f + image[z+ny-2] * 0.1f;

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

int mod(int x, int n) {
  return (x + n) % n;
}