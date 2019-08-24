#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STEPS       100                /* number of time steps */
#define MAXWORKER   8                  /* maximum number of worker tasks */
#define MINWORKER   3                  /* minimum number of worker tasks */
#define BEGIN       1                  /* message tag */
#define NONE        -1                  /* indicates no neighbor */
#define DONE        4                  /* message tag */
#define MASTER      0                  /* taskid of first process */

//New Params
#define BLOCK 100
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */
#define UTAG        5                  /* message tag */
#define DTAG        6                  /* message tag */

struct Parms {
  float cx;
  float cy;
} parms = {0.1, 0.1};

int main (int argc, char *argv[]){

void inidat(), prtdat(), update(), update_hv(), firstAndLast();
float  ***u;        /* array for grid */
int	taskid,                     /* this task's unique id */
	numworkers,                 /* number of worker processes */
	numtasks,                   /* number of tasks */
	averow,rows,offset,extra,   /* for sending rows of data */
	dest, source,               /* to - from for message send-receive */
	msgtype,                    /* for message types */
	rc,start,end,               /* misc */
	i,ix,iy,iz,it,j;              /* loop variables */
MPI_Status status;

//new vars
int left, right, up, down;       /* neighbor tasks */
int start_h, end_h, start_v, end_v;
MPI_Request left_r, right_r, up_r, down_r;
MPI_Datatype MPI_row,MPI_column;

int row=0;
double start_time=0.0,end_time=0.0,task_time=0.0,reduced_time=0.0;
int checkboard = BLOCK;

/* First, find out my taskid and how many tasks are running */
   MPI_Init(&argc,&argv);
   MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
   MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
   if(sqrt(numtasks)!=floor(sqrt(numtasks))){
      printf("We must have an equal number of blocks(ex 3x3,4x4)\n");
      MPI_Abort(MPI_COMM_WORLD,rc);
      exit(1);
   }
      /* allocate memory for block */
      u=(float***)malloc(2*sizeof(float**));
      for(i=0;i<2;i++){
        u[i]=(float**)malloc((BLOCK+2)*sizeof(float*));
        for(j=0;j<(BLOCK+2);j++){
          u[i][j]=(float*)malloc((BLOCK+2)*sizeof(float));
        }
      }
      /* Initialize everything - including the borders - to zero */
      for (iz=0; iz<2; iz++)
         for (ix=0; ix<BLOCK+2; ix++)
            for (iy=0; iy<BLOCK+2; iy++)
               u[iz][ix][iy] = 0.0;
      /* Initialize block to random values */
      inidat(BLOCK+2,BLOCK+2,u[0]);

      /* Calculate neighboors */
      row=(int)sqrt(numtasks);

      /* up */
      if(taskid<row){
        start_h=2;
        up=NONE;
      }
      else{
        start_h=1;
        up=taskid-row;
      }

      /* down */
      if(taskid>=(numtasks-row)){
        end_h=BLOCK-1;
        down=NONE;
      }
      else{
        end_h=BLOCK;
        down=taskid+row;
      }

      /* left */
      if((taskid%row)==0){
        start_v=2;
        left=NONE;
      }
      else{
        start_v=1;
        left=taskid-1;
      }

      /* right */
      if((taskid%row)==(row-1)){
        end_v=BLOCK-1;
        right=NONE;
      }
      else{
        end_v=BLOCK;
        right=taskid+1;
      }

      printf("for %d task id: UP=%d DOWN=%d LEFT=%d RIGHT=%d\n",taskid,up,down,left,right);

      MPI_Type_vector(BLOCK,1,BLOCK+2,MPI_FLOAT,&MPI_column);
      MPI_Type_commit(&MPI_column);
      MPI_Type_vector(BLOCK,1,BLOCK+2,MPI_FLOAT,&MPI_row);
      MPI_Type_commit(&MPI_row);
      // REMOVE THIS WHEN YOU ARE DONE WITH THE loop
      /* -----
      */
      MPI_Finalize();
      /* -----
      */
      start_time=MPI_Wtime();
      /* for loop */
      iz = 0;
      for (it = 1; it <= STEPS; it++)
      {
         //if up exists then send to X and receive from X
         if (left != NONE)
         {
            //printf("left\n");
            MPI_Isend(&u[iz][1][1], BLOCK, MPI_FLOAT, left, RTAG, MPI_COMM_WORLD, &left_r);
            source = left;
            msgtype = LTAG;
            MPI_Irecv(&u[iz][1][0], BLOCK, MPI_FLOAT, source, msgtype, MPI_COMM_WORLD, &left_r);
         }
         if (right != NONE)
         {
            //printf("right\n");
            MPI_Isend(&u[iz][1][BLOCK], BLOCK, MPI_FLOAT, right, LTAG, MPI_COMM_WORLD, &right_r);
            source = right;
            msgtype = RTAG;
            MPI_Irecv(&u[iz][1][BLOCK], BLOCK, MPI_FLOAT, source, msgtype, MPI_COMM_WORLD, &right_r);
         }
         if (up != NONE)
         {
            //printf("up\n");
            MPI_Isend(&u[iz][1][1], BLOCK, MPI_FLOAT, up, DTAG, MPI_COMM_WORLD, &up_r);
            source = up;
            msgtype = UTAG;
            MPI_Irecv(&u[iz][0][1], BLOCK, MPI_FLOAT, source, msgtype, MPI_COMM_WORLD, &up_r);
         }
         if (down != NONE)
         {
            //printf("down\n");
            MPI_Isend(&u[iz][BLOCK][1], BLOCK, MPI_FLOAT, down, UTAG, MPI_COMM_WORLD, &down_r);
            source = down;
            msgtype = DTAG;
            MPI_Irecv(&u[iz][BLOCK][1], BLOCK, MPI_FLOAT, source, msgtype, MPI_COMM_WORLD, &down_r);
         }

        //update(start,end,BLOCK,&u[iz][0][0],&u[1-iz][0][0]);
        checkboard = BLOCK + 2;
        update_hv(start_h, start_v, end_h, end_v, checkboard, &u[iz][0][0], &u[1-iz][0][0]);

        if (left != NONE)
          MPI_Wait(&left_r, MPI_STATUS_IGNORE);
        if (right != NONE)
          MPI_Wait(&right_r, MPI_STATUS_IGNORE);
        if (up != NONE)
          MPI_Wait(&up_r, MPI_STATUS_IGNORE);
        if (down != NONE)
          MPI_Wait(&down_r, MPI_STATUS_IGNORE);

        firstAndLast(checkboard, start_h, start_v, end_h, end_v, checkboard, &u[iz][0][0], &u[1-iz][0][0]);
        
        if (left != NONE)
          MPI_Wait(&left_r, MPI_STATUS_IGNORE);
        if (right != NONE)
          MPI_Wait(&right_r, MPI_STATUS_IGNORE);
        if (up != NONE)
          MPI_Wait(&up_r, MPI_STATUS_IGNORE);
        if (down != NONE)
          MPI_Wait(&down_r, MPI_STATUS_IGNORE);

         iz = 1 - iz;
      }
      end_time=MPI_Wtime();
      task_time=start_time-end_time;
      /* free data */
      MPI_Type_free(&MPI_column);
      MPI_Type_free(&MPI_row);
      for(i=0;i<BLOCK+2;i++){
        free(u[0][i]);
        free(u[1][i]);
      }
      free(u[0]);
      free(u[1]);
      free(u);
      MPI_Finalize();
}



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

/**************************************************************************
 *  subroutine update for both orientations
 ****************************************************************************/
void update_hv(int start_h, int start_v, int end_h, int end_v, int ny, float *u1, float *u2)
{
   int ix, iy;
   for (ix = start_h; ix <= end_h; ix++)
      for (iy = start_v; iy <= end_v; iy++)
         *(u2+ix*ny+iy) = *(u1+ix*ny+iy)  +
                          parms.cx * (*(u1+(ix+1)*ny+iy) +
                          *(u1+(ix-1)*ny+iy) -
                          2.0 * *(u1+ix*ny+iy)) +
                          parms.cy * (*(u1+ix*ny+iy+1) +
                         *(u1+ix*ny+iy-1) -
                          2.0 * *(u1+ix*ny+iy));
}

/*calculate first and last rows and columns*/
void firstAndLast(int checkboard, int start_h, int start_v, int end_h, int end_v, int ny, float *u1, float *u2)
{
  update_hv(start_h + 1, start_v, end_h - 1, start_v, checkboard, u1, u2);
  update_hv(start_h + 1, end_v, end_h - 1, end_v, checkboard, u1, u2);
  update_hv(start_h, start_v, start_h, end_v, checkboard, u1, u2);
  update_hv(end_h, start_v, end_h, end_v, checkboard, u1, u2);
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, float **u) {
int ix, iy;
for (ix = 0; ix < nx; ix++)
  for (iy = 0; iy < ny; iy++)
     u[ix][iy] = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float *u1, char *fnam) {
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