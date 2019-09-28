/****************************************************************************
 * FILE: mpi_heat2D.c
 * DESCRIPTIONS:
 *   HEAT2D Example - Parallelized C Version
 *   This example is based on a simplified two-dimensional heat
 *   equation domain decomposition.  The initial temperature is computed to be
 *   high in the middle of the domain and zero at the boundaries.  The
 *   boundaries are held at zero throughout the simulation.  During the
 *   time-stepping, an array containing two domains is used; these domains
 *   alternate between old data and new data.
 *
 *   In this parallelized version, the grid is decomposed by the master
 *   process and then distributed by rows to the worker processes.  At each
 *   time step, worker processes must exchange border data with neighbors,
 *   because a grid point's current temperature depends upon it's previous
 *   time step value plus the values of the neighboring grid points.  Upon
 *   completion of all time steps, the worker processes return their results
 *   to the master process.
 *
 *   Two data files are produced: an initial data set and a final data set.
 * AUTHOR: Blaise Barney - adapted from D. Turner's serial C version. Converted
 *   to MPI: George L. Gusciora (1/95)
 * LAST REVISED: 04/02/05
 ****************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NXPROB      20                 /* x dimension of problem grid */
#define NYPROB      20                 /* y dimension of problem grid */
#define STEPS       1000                /* number of time steps */
#define MAXWORKER   8                  /* maximum number of worker tasks */
#define MINWORKER   3                  /* minimum number of worker tasks */
#define BEGIN       1                  /* message tag */
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */
#define NONE        0                  /* indicates no neighbor */
#define DONE        4                  /* message tag */
#define MASTER      0                  /* taskid of first process */

#define BLOCK_H   5
#define BLOCK_V   5
#define THREADS   32

struct Parms {
  float cx;
  float cy;
} parms = {0.1, 0.1};

/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void update(int start, int end, int ny, float *u1, float *u2)
{
   int ix, iy;
   for (ix = start; ix <= end; ix++)
      for (iy = 1; iy <= ny-2; iy++)
         *(u2+ix*ny+iy) = *(u1+ix*ny+iy)  +
                          parms.cx * (*(u1+(ix+1)*ny+iy) +
                          *(u1+(ix-1)*ny+iy) -
                          2.0 * *(u1+ix*ny+iy)) +
                          parms.cy * (*(u1+ix*ny+iy+1) +
                         *(u1+ix*ny+iy-1) -
                          2.0 * *(u1+ix*ny+iy));
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, float *u) {
int ix, iy;
for (ix = 0; ix <= nx-1; ix++)
  for (iy = 0; iy <= ny-1; iy++)
     {*(u+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
     if (*(u+ix*ny+iy) > 10000.0)
     printf("%f\n", *(u+ix*ny+iy));}
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float *u1, const char *fnam) {
int ix, iy;
FILE *fp;

fp = fopen(fnam, "w");
for (iy = ny-1; iy >= 0; iy--) {
  for (ix = 0; ix <= nx-1; ix++) {
    fprintf(fp, "%6.1f", *(u1+ix*ny+iy));
    if (ix != nx-1)
      fprintf(fp, " ");
    else
      fprintf(fp, "\n");
    }
  }
fclose(fp);
}

__global__ void cuda_update(float *u0, float *u1)
{
  int ix, iy;
  ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
  iy = blockIdx.y * blockDim.y + threadIdx.y + 1;

  if (ix > 0 && iy > 0)
  {
    if (ix + iy < NXPROB + NYPROB - 2)
    {
      *(u1+ix*NYPROB+iy) = *(u0+ix*NYPROB+iy)  +
                          0.1f * (*(u0+(ix+1)*NYPROB+iy) +
                          *(u0+(ix-1)*NYPROB+iy) -
                          2.0 * *(u0+ix*NYPROB+iy)) +
                          0.1f * (*(u0+ix*NYPROB+iy+1) +
                         *(u0+ix*NYPROB+iy-1) -
                          2.0 * *(u0+ix*NYPROB+iy));
    }
  }
}

int main (int argc, char *argv[])
{
   int i;
   float *u;
   float *cuda_u0, *cuda_u1;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   float ms = 0.0;

   dim3 dimBlocks(BLOCK_H, BLOCK_V);
   dim3 dimThreads((NXPROB / BLOCK_H) + ((NXPROB % BLOCK_H) != 0), (NYPROB / BLOCK_V) + ((NYPROB % BLOCK_V) != 0));

   //malloc host 
   u = (float*)malloc(NXPROB*NYPROB*sizeof(float));

   //malloc device
   cudaMalloc((void**)&cuda_u0, (NXPROB*NYPROB*sizeof(float)));
   cudaMalloc((void**)&cuda_u1, (NXPROB*NYPROB*sizeof(float)));

   inidat(NXPROB, NYPROB, u); //initialize
   prtdat(NXPROB, NYPROB, u, "initial.dat"); //print

   //copy from host to device
   cudaMemcpy(cuda_u0, u, (NXPROB*NYPROB*sizeof(float)), cudaMemcpyHostToDevice);
   cudaMemcpy(cuda_u1, u, (NXPROB*NYPROB*sizeof(float)), cudaMemcpyHostToDevice);

   cudaEventRecord(start);
   for (i = 0; i < STEPS; i++)
      if (i%2 == 0)  {cuda_update<<<dimBlocks, dimThreads>>>(cuda_u0, cuda_u1);}
      else  {cuda_update<<<dimBlocks, dimThreads>>>(cuda_u1, cuda_u0);}
   cudaEventRecord(stop);

   //copy from device to host
   if (STEPS%2 == 0) {cudaMemcpy(u, cuda_u0, (NXPROB*NYPROB*sizeof(float)), cudaMemcpyDeviceToHost);}
   else {cudaMemcpy(u, cuda_u1, (NXPROB*NYPROB*sizeof(float)), cudaMemcpyDeviceToHost);}
   
   prtdat(NXPROB, NYPROB, u, "final.dat");   //print
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&ms, start, stop);
   printf("Time: %4.10f ms\n", ms);

   cudaFree(cuda_u0);
   cudaFree(cuda_u1);
   free(u);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   return 0;
}
