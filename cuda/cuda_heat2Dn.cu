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
#define STEPS       10                /* number of time steps */
#define MAXWORKER   8                  /* maximum number of worker tasks */
#define MINWORKER   3                  /* minimum number of worker tasks */
#define BEGIN       1                  /* message tag */
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */
#define NONE        0                  /* indicates no neighbor */
#define DONE        4                  /* message tag */
#define MASTER      0                  /* taskid of first process */

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

int main (int argc, char *argv[])
{
   int i;
   float *u;
   float *cuda_u0, *cuda_u1;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   float ms = 0.0;

   //malloc host 
   u = (float*)malloc(NXPROB*NYPROB*sizeof(float));

   //malloc device
   cudaMalloc((void**)&cuda_u0, (NXPROB*NYPROB*sizeof(float)));
   cudaMalloc((void**)&cuda_u1, (NXPROB*NYPROB*sizeof(float)));

   //initialize
   inidat(NXPROB, NYPROB, u);
   //print
   prtdat(NXPROB, NYPROB, u, "initial.dat");

   //copy from host to device
   cudaMemcpy(cuda_u0, u, (NXPROB*NYPROB*sizeof(float)), cudaMemcpyHostToDevice);
   cudaMemcpy(cuda_u1, u, (NXPROB*NYPROB*sizeof(float)), cudaMemcpyHostToDevice);

   cudaEventRecord(start);
   for (i = 0; i < STEPS; i++)
   {

   }
   cudaEventRecord(stop);


   //copy from device to host   
   cudaMemcpy(u, cuda_u0, (NXPROB*NYPROB*sizeof(float)), cudaMemcpyDeviceToHost);

   //print
   prtdat(NXPROB, NYPROB, u, "final.dat");
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&ms, start, stop);
   printf("ms: %4.10f\n", ms);

   cudaFree(cuda_u0);
   cudaFree(cuda_u1);
   free(u);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   return 0;
}
