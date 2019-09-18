#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define NXPROB      20                 /* x dimension of problem grid */
#define NYPROB      20                 /* y dimension of problem grid */
#define STEPS       100                /* number of time steps */
#define MAXWORKER   8                  /* maximum number of worker tasks */
#define MINWORKER   3                  /* minimum number of worker tasks */
#define BEGIN       1                  /* message tag */
#define NONE        -1                  /* indicates no neighbor */
#define DONE        4                  /* message tag */
#define MASTER      0                  /* taskid of first process */

//New Params
#define LTAG        2                  /* message tag */
#define RTAG        3                  /* message tag */
#define UTAG        5                  /* message tag */
#define DTAG        6                  /* message tag */
#define MAX_TEMP    500
#define MIN_TEMP    10

struct Parms {
  float cx;
  float cy;
} parms = {0.1, 0.1};

int main (int argc, char *argv[]){

void inidat(), prtdat(), update(), update_hv(), firstAndLast(), inidat_block();
float  ***u;        /* array for grid */
int	taskid,                     /* this task's unique id */
	numworkers,                 /* number of worker processes */
	numtasks,                   /* number of tasks */
	averow,rows,offset,extra,   /* for sending rows of data */
	dest, source,               /* to - from for message send-receive */
	msgtype,                    /* for message types */
	rc,start,end,               /* misc */
	i,ix,iy,iz,it,j,k;              /* loop variables */
MPI_Status status;

//new vars
//Comm_world
MPI_Comm comm_cart;
int ndims;
int dims[ndims];
int periods[ndims];
int reorder;

float *line[2];
int left, right, up, down;       /* neighbor tasks */
int start_h, end_h, start_v, end_v;
MPI_Request Sleft_r, Sright_r, Sup_r, Sdown_r;  //send
MPI_Request Rleft_r, Rright_r, Rup_r, Rdown_r;  //receive
MPI_Datatype MPI_row,MPI_column;

int row=0;
double start_time=0.0,end_time=0.0,task_time=0.0,reduced_time=0.0;
int BLOCK, checkboard;

  //--------------------------------------------------------------
  // Find out taskid and how many tasks are running
  //--------------------------------------------------------------

   MPI_Init(&argc,&argv);
   MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
   MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

    //--------------------------------------------------------------
    // Calculate number of tasks in each row
    //--------------------------------------------------------------
    row=(int)sqrt(numtasks);
    if(sqrt(numtasks)!=floor(sqrt(numtasks))){
      printf("We must have an equal number of blocks(ex 3x3,4x4)\n");
      MPI_Abort(MPI_COMM_WORLD,rc);
      exit(1);
    }
    else{
      BLOCK = (int)(sqrt(NXPROB * NYPROB / numtasks)); //get size of BLOCK
    }
      //--------------------------------------------------------------
      // Allocate memory for block
      //--------------------------------------------------------------

      u = (float***)malloc(2*sizeof(float**));
      u[0] = (float**)malloc((BLOCK+2)*sizeof(float*));
      //malloc in a line so column stride can work
      line[0] = (float*)malloc((BLOCK+2)*(BLOCK+2)*sizeof(float));
      for(j = 0; j < BLOCK + 2; j++)
          u[0][j] = j*(BLOCK+2) + line[0];
      u[1] = (float**)malloc((BLOCK+2)*sizeof(float*));
      //malloc in a line so column stride can work
      line[1] = (float*)malloc((BLOCK+2)*(BLOCK+2)*sizeof(float));
      for(j = 0; j < BLOCK + 2; j++)
          u[1][j] = j*(BLOCK+2) + line[1];

     //--------------------------------------------------------------
     // Initialize everything - including the borders - to zero
     //--------------------------------------------------------------
     //Y orizontia   //X katheta

      for (iz=0; iz<2; iz++)
        for (ix=0; ix<BLOCK+2; ix++)
          for (iy=0; iy<BLOCK+2; iy++)
            u[iz][ix][iy] = 0.0;

      //--------------------------------------------------------------
      // Initialize block to random values
      //--------------------------------------------------------------

      inidat_block(BLOCK+2,BLOCK+2,u[0], taskid, numtasks);

      char str[50];
      sprintf(str, "%d", taskid);
      strcat(str, "initial.dat");
      prtdat(BLOCK + 1, BLOCK + 1, u[0], str);

////////////////////////////////////////////////////////////////////////////

      ndims = 2;
      reorder = 0;
      for (i = 0; i < ndims; i++) { periods[i] = 0; dims[i] = row;}
      MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);

      //--------------------------------------------------------------
      //Calculate neighbor task ids (if they exist)
      //--------------------------------------------------------------

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

      //--------------------------------------------------------------
      //                Datatypes initialization
      //--------------------------------------------------------------

      MPI_Type_vector(BLOCK + 2, 1, 1, MPI_FLOAT, &MPI_row);
      MPI_Type_commit(&MPI_row);
      MPI_Type_vector(BLOCK + 2, 1, BLOCK + 2, MPI_FLOAT, &MPI_column);
      MPI_Type_commit(&MPI_column);

            //--------------------------------------------------------------
      //                Cartesian communicator
      //--------------------------------------------------------------
      //http://mpi.deino.net/mpi_functions/MPI_Cart_shift.html

      MPI_Cart_shift(comm_cart, 0, 1, &up, &down);
      MPI_Cart_shift(comm_cart, 1, 1, &left, &right);
      printf("for %d task id: UP=%d DOWN=%d LEFT=%d RIGHT=%d\n",taskid,up,down,left,right);
      iz = 0;
      if (left <= NONE)  left = MPI_PROC_NULL;
      if (right <= NONE)  right = MPI_PROC_NULL;
      if (up <= NONE)  up = MPI_PROC_NULL;
      if (down <= NONE)  down = MPI_PROC_NULL;

      //--------------------------------------------------------------
      //               MPI Send - Recv - Start
      //--------------------------------------------------------------

      MPI_Send_init(&u[iz][0][1], 1, MPI_column, left, RTAG, comm_cart, &Sleft_r);
      MPI_Send_init(&u[iz][0][BLOCK], 1, MPI_column, right, LTAG, comm_cart, &Sright_r);
      MPI_Send_init(&u[iz][1][0], 1, MPI_row, up, DTAG, comm_cart, &Sup_r);
      MPI_Send_init(&u[iz][BLOCK][0], 1, MPI_row, down, UTAG, comm_cart, &Sdown_r);
      
      MPI_Recv_init(&u[iz][0][0], 1, MPI_column, left, LTAG, comm_cart, &Rleft_r);
      MPI_Recv_init(&u[iz][0][BLOCK+1], 1, MPI_column, right, RTAG, comm_cart, &Rright_r);
      MPI_Recv_init(&u[iz][0][0], 1, MPI_row, up, UTAG, comm_cart, &Rup_r);
      MPI_Recv_init(&u[iz][BLOCK+1][0], 1, MPI_row, down, DTAG, comm_cart, &Rdown_r);


      MPI_Start(&Sleft_r);
      MPI_Start(&Sright_r); 
      MPI_Start(&Sup_r); 
      MPI_Start(&Sdown_r); 
      MPI_Start(&Rleft_r); 
      MPI_Start(&Rright_r); 
      MPI_Start(&Rup_r); 
      MPI_Start(&Rdown_r);

      //--------------------------------------------------------------
      //                for loop
      //--------------------------------------------------------------
      checkboard=BLOCK+2;
      MPI_Barrier(comm_cart);
      start_time=MPI_Wtime();
      for (it = 1; it <= STEPS; it++)
      {
        //--------------------------------------------------------------
        //                Send-Receive
        //--------------------------------------------------------------

        //--------------------------------------------------------------
        //                left
        //--------------------------------------------------------------

        MPI_Irecv(&u[iz][0][0], 1, MPI_column, left, LTAG, comm_cart, &Rleft_r);
        MPI_Isend(&u[iz][0][1], 1, MPI_column, left, RTAG, comm_cart, &Sleft_r);

        //--------------------------------------------------------------
        //                right
        //--------------------------------------------------------------

        MPI_Irecv(&u[iz][0][BLOCK+1], 1, MPI_column, right, RTAG, comm_cart, &Rright_r);
        MPI_Isend(&u[iz][0][BLOCK], 1, MPI_column, right, LTAG, comm_cart, &Sright_r);

        //--------------------------------------------------------------
        //                up
        //--------------------------------------------------------------

        MPI_Irecv(&u[iz][0][0], 1, MPI_row, up, UTAG, comm_cart, &Rup_r);
        MPI_Isend(&u[iz][1][0], 1, MPI_row, up, DTAG, comm_cart, &Sup_r);

        //--------------------------------------------------------------
        //                down
        //--------------------------------------------------------------

        MPI_Irecv(&u[iz][BLOCK+1][0], 1, MPI_row, down, DTAG, comm_cart, &Rdown_r);
        MPI_Isend(&u[iz][BLOCK][0], 1, MPI_row, down, UTAG, comm_cart, &Sdown_r);


        //--------------------------------------------------------------
        //                Calculate white spaces
        //--------------------------------------------------------------

        update_hv(start_h + 1, start_v + 1, end_h - 1, end_v - 1, checkboard, u[iz], u[1-iz]);

        //--------------------------------------------------------------
        //                Wait for all
        //--------------------------------------------------------------

          MPI_Wait(&Rleft_r, MPI_STATUS_IGNORE);
          MPI_Wait(&Rright_r, MPI_STATUS_IGNORE);
          MPI_Wait(&Rup_r, MPI_STATUS_IGNORE);
          MPI_Wait(&Rdown_r, MPI_STATUS_IGNORE);

        //--------------------------------------------------------------
        //                Calculate for all
        //--------------------------------------------------------------

        firstAndLast(checkboard, start_h, start_v, end_h, end_v, checkboard, u[iz], u[1-iz]);

        //--------------------------------------------------------------
        //                Wait for all
        //--------------------------------------------------------------
        
          MPI_Wait(&Sleft_r, MPI_STATUS_IGNORE);
          MPI_Wait(&Sright_r, MPI_STATUS_IGNORE);
          MPI_Wait(&Sup_r, MPI_STATUS_IGNORE);
          MPI_Wait(&Sdown_r, MPI_STATUS_IGNORE);

          //--------------------------------------------------------------
          //                Next loop
          //--------------------------------------------------------------

         iz = 1 - iz;
      }
      //--------------------------------------------------------------
      //               Loop ends here
      //--------------------------------------------------------------
      //--------------------------------------------------------------
      //               Calculate elapsed time
      //--------------------------------------------------------------

      end_time=MPI_Wtime();
      task_time = end_time - start_time;

      //--------------------------------------------------------------
      //                Print data
      //--------------------------------------------------------------

      sprintf(str, "%d", taskid);
      strcat(str, "final.dat");
      prtdat(BLOCK + 1, BLOCK + 1, u[0], str);

      //--------------------------------------------------------------
      //                Free datatypes
      //--------------------------------------------------------------

      MPI_Type_free(&MPI_column);
      MPI_Type_free(&MPI_row);

      //--------------------------------------------------------------
      //                Free allocated memory
      //--------------------------------------------------------------

      free(line[1]);
      free(line[0]);
      free(u[1]);
      free(u[0]);
      free(u);
      /*MPI_Request_free(&Sleft_r);
      MPI_Request_free(&Sright_r); 
      MPI_Request_free(&Sup_r); 
      MPI_Request_free(&Sdown_r); 
      MPI_Request_free(&Rleft_r); 
      MPI_Request_free(&Rright_r); 
      MPI_Request_free(&Rup_r); 
      MPI_Request_free(&Rdown_r);*/
 

      //--------------------------------------------------------------
      //                End of each task
      //--------------------------------------------------------------
      printf("- MPI_Finalize task id %d -\n", taskid);
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
void update_hv(int start_h, int start_v, int end_h, int end_v, int ny, float **u1, float **u2)
{
   int ix, iy;
   for (ix = start_h; ix <= end_h; ix++)
      for (iy = start_v; iy <= end_v; iy++)
      {
       /* printf("ix: %d iy: %d\n", ix, iy);
        printf("u1[ix][iy]: %6.1f\n", u1[ix][iy]);
        printf("u1[ix-1][iy]: %6.1f\n", u1[ix-1][iy]);
        printf("u1[ix+1][iy]: %6.1f\n", u1[ix+1][iy]);
        printf("u1[ix][iy-1]: %6.1f\n", u1[ix][iy-1]);
        printf("u1[ix][iy+1]: %6.1f\n", u1[ix][iy+1]);*/

        u2[ix][iy] = u1[ix][iy] +
                          parms.cx * (u1[ix+1][iy] +
                          u1[ix-1][iy] -
                          2.0 * u1[ix][iy]) +
                          parms.cy * (u1[ix][iy+1] +
                         u1[ix][iy-1] -
                          2.0 * u1[ix][iy]);
      }
}

/*calculate first and last rows and columns*/
void firstAndLast(int checkboard, int start_h, int start_v, int end_h, int end_v, int ny, float **u1, float **u2)
{
  //printf("%d %d %d %d\n", start_h, start_v + 1, start_h, end_v - 2);
  update_hv(start_h , start_v, end_h - 1, start_v, checkboard, u1, u2);
  update_hv(start_h , end_v - 1, end_h - 1, end_v - 1, checkboard, u1, u2);
  update_hv(start_h, start_v + 1, start_h, end_v - 2, checkboard, u1, u2);
  update_hv(end_h - 1, start_v, end_h - 1, end_v - 1, checkboard, u1, u2);
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/
void inidat(int nx, int ny, float *u)
{
int ix, iy;
for (ix = 0; ix <= nx-1; ix++)
  for (iy = 0; iy <= ny-1; iy++)
     *(u+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
}

void inidat_block(int nx, int ny, float **u, int taskid, int tasks) { //init in relation with taskid because we dont want all of the block starts-ends to be 0
int ix, iy;
int startx = 0, starty = 0;

for (ix = startx; ix < nx; ix++)
  for (iy = starty; iy < ny; iy++)
    {
      u[ix][iy] = (float)((ix * (nx - ix - 1) * iy * (ny - iy - 1)));
    //  if (u[ix][iy] != 0.0)
    //    u[ix][iy] = 1.1;
      /*if (taskid == 0)
      u[ix][iy] = ((float)ix+(float)((float)iy/100.0));
      else
      u[ix][iy] = (float)taskid;*/

    }
//every block will have 0.0 at each border. 0.0 will be kept the same for blocks with no neighbors
//or the neighboring column/row will replace it
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float **u1, char *fnam) {
int ix, iy;
FILE *fp;

fp = fopen(fnam, "w");
for (ix = 0; ix <= nx; ix++) {
  for (iy = 0; iy <= ny; iy++) {
    //fprintf(fp, "%p", &u1[ix][iy]);
    fprintf(fp, "%6.2f", u1[ix][iy]);
    if (iy != ny)
      fprintf(fp, " ");
    else
      fprintf(fp, "\n");
    }
  }
fclose(fp);
}
