#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// run with: mpirun -np 4 ./stencil 1024 1024 100

// Define output file name
#define OUTPUT_FILE "stencil.pgm"

#define MASTER 0
#define NDIMS 2

void stencil(const int z, const int nx, const int ny, const int numy, float *  image, float *  tmp_image);
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

  // Prepare image
  float *image = malloc(sizeof(float)*numx*numy);
  float *tmp_image = malloc(sizeof(float)*numx*numy);
  init_image(numx, numy, image, tmp_image);

  // Set up MPI
  int rank, size, N, E, S, W;
  int flag, tag = 0, reorder = 0;                 // silly flags
  int dims[NDIMS], periods[NDIMS], coords[NDIMS];
  MPI_Comm COMM_CART;
  MPI_Status status;

  MPI_Init( &argc, &argv );
  MPI_Initialized(&flag);
  if (flag != 1)
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );

  // Set up cartesian grid
  if (size % 2 != 0) {
    fprintf(stderr, "even cohort size required\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  for (int i = 0; i < NDIMS; i++) {
    dims[i] = 0;
    periods[i] = 1;
  }
  MPI_Dims_create(size, NDIMS, dims);
  if (rank == MASTER)
    printf("%d nodes, dims [%d,%d], %d %d %d\n", size, dims[0], dims[1], numx, numy, niters);
  MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &COMM_CART);

  // get coords, and rank of neighbours
  MPI_Cart_coords(COMM_CART, rank, NDIMS, coords);
  MPI_Cart_shift(COMM_CART, 0, 1, &W, &E);
  MPI_Cart_shift(COMM_CART, 1, 1, &S, &N);

  // determine portion width + height
  int nx = numx / dims[0];
  int ny = numy / dims[1];

  // index of top left cell of portion
  int col = rank / dims[1];
  int row = dims[1] - (rank % dims[1]) - 1;
  int z = nx * col * numy + row * ny;

  // account for input sizes not divisible by cohort size
  if (numx % dims[0] != 0) {
    if (col < numx % dims[0])
      nx++;
    z += numy*((col > numx % dims[0]) ? numx % dims[0] : col);
  }
  if (numy % dims[1] != 0) {
    if (row < numy % dims[1])
      ny++;
    z += (row > numy % dims[1]) ? numy % dims[1] : row;
  }

  // these indexes include halo columns
  int _nx = nx;
  int _ny = ny;
  int _z = z;

  // add 1 or two to nx depending on how many halo cols
  // also shift starting index back 1 col if not first
  if (col == 0) {
    // one halo to right
    _nx += 1;
  } else if (col == dims[0] - 1) {
    // one halo to left
    _nx += 1;
    _z  -= numy;
  } else {
    // halo to left and right
    _nx += 2;
    _z  -= numy;
  } if (row == 0) {
    // one halo below
    _ny += 1;
  } else if (row == dims[1] - 1) {
    // one halo above
    _ny += 1;
    _z  -= 1;
  } else {
    // halo above and below
    _ny += 2;
    _z  -= 1;
  }

  float haloN[nx]; // send buffers
  float haloE[ny];
  float haloS[nx];
  float haloW[ny];
  float haloR[(ny>nx)?ny:nx]; // recv buffer

  int z_h = (row == 0) ? _z : _z+1;
  int z_v = (col == 0) ? _z : _z+numy;

  // syncronise processes
  MPI_Barrier(MPI_COMM_WORLD);

  // Start timing my code
  double tic = wtime();

  for (int t = 0; t < niters; ++t) {
    // change tmp_image
    stencil(_z, _nx, _ny, numy, image, tmp_image);

    // populate halos
    for (int x = 0; x < nx; x++)
      haloS[x] = tmp_image[z+ny-1 +x*numy];   // south
    for (int x = 0; x < nx; x++)
      haloN[x] = tmp_image[z      +x*numy];   // north
    for (int y = 0; y < ny; y++)
      haloE[y] = tmp_image[z+(nx-1)*numy +y]; // east
    for (int y = 0; y < ny; y++)
      haloW[y] = tmp_image[z             +y]; // west

    // send horizontally
    if (col == 0) {
      // send east, receive from east
      MPI_Sendrecv(haloE, ny, MPI_FLOAT, E, tag,
	      haloR, ny, MPI_FLOAT, E, tag, MPI_COMM_WORLD, &status);

      // process received halo
      for (int y = 0; y < ny; y++)
        tmp_image[z_h+(_nx-1)*numy +y] = haloR[y];

    } else if (col == dims[0] - 1) {
      // send west, receive from west
      MPI_Sendrecv(haloW, ny, MPI_FLOAT, W, tag,
	      haloR, ny, MPI_FLOAT, W, tag, MPI_COMM_WORLD, &status);

      // process received halo
      for (int y = 0; y < ny; y++)
        tmp_image[z_h+y] = haloR[y];

    } else {
      // send east, receive from west
      MPI_Sendrecv(haloE, ny, MPI_FLOAT, E, tag,
	      haloR, ny, MPI_FLOAT, W, tag, MPI_COMM_WORLD, &status);

      // send west, receive from east
      MPI_Sendrecv(haloW, ny, MPI_FLOAT, W, tag,
	      haloE, ny, MPI_FLOAT, E, tag, MPI_COMM_WORLD, &status);

      // process halos
      for (int y = 0; y < ny; y++)
        tmp_image[z_h+y] = haloR[y];
      for (int y = 0; y < ny; y++)
        tmp_image[z_h+(_nx-1)*numy +y] = haloE[y];
    }

    // send vertically
    if (row == 0) {
      // send south, receive from south
      MPI_Sendrecv(haloS, nx, MPI_FLOAT, S, tag,
	      haloR, nx, MPI_FLOAT, S, tag, MPI_COMM_WORLD, &status);

      // process received halo
      for (int x = 0; x < nx; x++)
        tmp_image[z_v+_ny-1 +x*numy] = haloR[x];

    } else if (row == dims[1] - 1) {
      // send north, receive from north
      MPI_Sendrecv(haloN, nx, MPI_FLOAT, N, tag,
	      haloR, nx, MPI_FLOAT, N, tag, MPI_COMM_WORLD, &status);

      // process received halo
      for (int x = 0; x < nx; x++)
        tmp_image[z_v +x*numy] = haloR[x];

    } else {
      // send north, receive from south
      MPI_Sendrecv(haloN, nx, MPI_FLOAT, N, tag,
	      haloR, nx, MPI_FLOAT, S, tag, MPI_COMM_WORLD, &status);
      // send south, receive from north
      MPI_Sendrecv(haloS, nx, MPI_FLOAT, S, tag,
	      haloN, nx, MPI_FLOAT, N, tag, MPI_COMM_WORLD, &status);

      // process halos
      for (int x = 0; x < nx; x++)
        tmp_image[z_v+_ny-1 +x*numy] = haloR[x];
      for (int x = 0; x < nx; x++)
        tmp_image[z_v +x*numy] = haloN[x];
    }

    // change image
    stencil(_z, _nx, _ny, numy, tmp_image, image);

    // populate halos
    for (int x = 0; x < nx; x++)
      haloS[x] = image[z+ny-1 +x*numy];   // south
    for (int x = 0; x < nx; x++)
      haloN[x] = image[z      +x*numy];   // north
    for (int y = 0; y < ny; y++)
      haloE[y] = image[z+(nx-1)*numy +y]; // east
    for (int y = 0; y < ny; y++)
      haloW[y] = image[z             +y]; // west

    // send horizontally
    if (col == 0) {
      // send east, receive from east
      MPI_Sendrecv(haloE, ny, MPI_FLOAT, E, tag,
	      haloR, ny, MPI_FLOAT, E, tag, MPI_COMM_WORLD, &status);

      // process received halo
      for (int y = 0; y < ny; y++)
        image[z_h+(_nx-1)*numy +y] = haloR[y];

    } else if (col == dims[0] - 1) {
      // send west, receive from west
      MPI_Sendrecv(haloW, ny, MPI_FLOAT, W, tag,
	      haloR, ny, MPI_FLOAT, W, tag, MPI_COMM_WORLD, &status);

      // process received halo
      for (int y = 0; y < ny; y++)
        image[z_h+y] = haloR[y];

    } else {
      // send east, receive from west
      MPI_Sendrecv(haloE, ny, MPI_FLOAT, E, tag,
	      haloR, ny, MPI_FLOAT, W, tag, MPI_COMM_WORLD, &status);

      // send west, receive from east
      MPI_Sendrecv(haloW, ny, MPI_FLOAT, W, tag,
	      haloE, ny, MPI_FLOAT, E, tag, MPI_COMM_WORLD, &status);

      // process halos
      for (int y = 0; y < ny; y++)
        image[z_h+y] = haloR[y];
      for (int y = 0; y < ny; y++)
        image[z_h+(_nx-1)*numy +y] = haloE[y];
    }

    // send vertically
    if (row == 0) {
      // send south, receive from south
      MPI_Sendrecv(haloS, nx, MPI_FLOAT, S, tag,
	      haloR, nx, MPI_FLOAT, S, tag, MPI_COMM_WORLD, &status);

      // process received halo
      for (int x = 0; x < nx; x++)
        image[z_v+_ny-1 +x*numy] = haloR[x];

    } else if (row == dims[1] - 1) {
      // send north, receive from north
      MPI_Sendrecv(haloN, nx, MPI_FLOAT, N, tag,
	      haloR, nx, MPI_FLOAT, N, tag, MPI_COMM_WORLD, &status);

      // process received halo
      for (int x = 0; x < nx; x++)
        image[z_v +x*numy] = haloR[x];

    } else {
      // send north, receive from south
      MPI_Sendrecv(haloN, nx, MPI_FLOAT, N, tag,
	      haloR, nx, MPI_FLOAT, S, tag, MPI_COMM_WORLD, &status);
      // send south, receive from north
      MPI_Sendrecv(haloS, nx, MPI_FLOAT, S, tag,
	      haloN, nx, MPI_FLOAT, N, tag, MPI_COMM_WORLD, &status);

      // process halos
      for (int x = 0; x < nx; x++)
        image[z_v+_ny-1 +x*numy] = haloR[x];
      for (int x = 0; x < nx; x++)
        image[z_v +x*numy] = haloN[x];
    }
  }

  // Stop timing my code
  double toc = wtime();

  float buffer[nx*ny];

  // combine at master
  if (rank == MASTER) {
    for (int src = 1; src < size; src++) {
      // determine portion width + height
      int nx_s = numx / dims[0];
      int ny_s = numy / dims[1];

      // index of top left cell of portion
      int col_s = src / dims[1];
      int row_s = dims[1] - (src % dims[1]) - 1;
      int z_s = nx_s * col_s * numy + row_s * ny_s;

      // account for input sizes not divisible by cohort size
      if (numx % dims[0] != 0) {
        if (col_s < numx % dims[0])
          nx_s++;
        z_s += numy*((col_s > numx % dims[0]) ? numx % dims[0] : col_s);
      }
      if (numy % dims[1] != 0) {
        if (row_s < numy % dims[1])
          ny_s++;
        z_s += (row_s > numy % dims[1]) ? numy % dims[1] : row_s;
      }

      MPI_Recv(buffer, nx_s*ny_s, MPI_FLOAT, src, tag, MPI_COMM_WORLD, &status);
      for (int i = 0; i < nx_s; i++)
        for (int j = 0; j < ny_s; j++)
          image[z_s +j +i*numy] = buffer[i*ny_s+j];
    }
  } else {
    for (int i = 0; i < nx; i++)
      for (int j = 0; j < ny; j++)
        buffer[i*ny+j] = image[z+j + i*numy];
    MPI_Send(buffer, nx*ny, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
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
 * numy height of img
 */
void stencil(const int z, const int nx, const int ny, const int numy, float * restrict image, float * restrict tmp_image) {

  // top left corner
  tmp_image[z] = image[z] * 0.6f + image[z+numy] * 0.1f + image[z+1] * 0.1f;

  // 'left' vertical edge cells
  for (int j = 1; j != ny-1; ++j) {  // VECTORIZING
    // i = 0
    tmp_image[z+j] =  image[z+j] * 0.6f + image[z+j+numy] * 0.1f + image[z+j-1] * 0.1f + image[z+j+1] * 0.1f;
  }

  // bottom left corner
  tmp_image[z+ny-1] = image[z+ny-1] * 0.6f + image[z+ny-1+numy] * 0.1f + image[z+ny-2] * 0.1f;

  // center columns
  for (int i = 1; i != nx-1; ++i) {
    // j = 0
    tmp_image[z+i*numy] = image[z+i*numy] * 0.6f + image[z+(i-1)*numy] * 0.1f + image[z+(i+1)*numy] * 0.1f + image[z+1+i*numy] * 0.1f;

    for (int j = 1; j != ny-1; ++j) {  // VECTORIZING
      tmp_image[z+j+i*numy] = image[z+j+i*numy] * 0.6f + image[z+j+(i-1)*numy] * 0.1f + image[z+j+(i+1)*numy] * 0.1f + image[z+j+i*numy-1] * 0.1f + image[z+j+i*numy+1] * 0.1f;
    }

    // j = (ny-1)
    tmp_image[z+ny-1+i*numy] = image[z+ny-1+i*numy] * 0.6f + image[z+ny-1+(i-1)*numy] * 0.1f + image[z+ny-1+(i+1)*numy] * 0.1f + image[z+ny-2+i*numy] * 0.1f;
  }

  // top right corner
  tmp_image[z+(nx-1)*numy] = image[z+(nx-1)*numy] * 0.6f + image[z+(nx-2)*numy] * 0.1f + image[z+1+(nx-1)*numy] * 0.1f;

  // 'right' vertical edge cells
  for (int j = 1; j != ny-1; ++j) {  // VECTORIZING
    // i = (nx-1)
    tmp_image[z+j+(nx-1)*numy] = image[z+j+(nx-1)*numy] * 0.6f + image[z+j+(nx-2)*numy] * 0.1f + image[z+j-1+(nx-1)*numy] * 0.1f + image[z+j+1+(nx-1)*numy] * 0.1f;
  }

  // bottom right corner
  tmp_image[z+ny-1+(nx-1)*numy] =  image[z+ny-1+(nx-1)*numy] * 0.6f + image[z+ny-2+(nx-1)*numy] * 0.1f + image[z+ny-1+(nx-2)*numy] * 0.1f;
  
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