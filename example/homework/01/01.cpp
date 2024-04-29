/*Problem 1 (30pts/24pts):
Write an MPI program that builds a 2D process topology of shape P ×Q. On each column of processes, store a
vector x of length M , distributed in a linear load-balanced fashion “vertically” (it will be replicated Q times).
Start with data only in process (0,0), and distribute it down the first column. Once it is distribute on column 0,
broadcast it horizontally in each process row. Allocate a vector y of length M that is replicated “horizontally”
in each process row and stored also in linear load-balanced distribution; there will be P replicas, one in each
process row. Using MPI Allreduce with the appropriate communicators, do the parallel
copy y := x. There should be P replicas of the answer in y when you’re done.
Notes:
• CSC4760: Leverage your work on Problem #0 above to help do this problem.
• CSC5760: Leverage your work on HW#2, Problem #5 to help do this problem
*/

#include "mpi.h"
#include "Distributions.h"
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int Q = 2;
    int P = world_size / Q;
    int M = 30;

    // First split based on ranks divided by Q
    int col_color = world_rank / Q;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, world_rank, &col_comm);

    // Second split based on ranks mod Q
    int row_color = world_rank % Q;
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, world_rank, &row_comm);

    int p = world_rank / Q;
    int q = world_rank % Q;
    LinearDistribution x_dist(P, M);
    LinearDistribution y_dist(Q, M);

    // Process (p,q) will have m elements of x
    int m = x_dist.m(p);
    // Process (p,q) will have n elements of y
    int n = y_dist.m(q);

    // Allocate the vectors x and y
    int *x_local = new int[m];
    for (int i = 0; i < m; i++) {
        x_local[i] = 0;
    }

    int *y_local = new int[n];
    for (int i = 0; i < n; i++) {
        y_local[i] = 0;
    }

    MPI_Request request;
    int *sendcounts = new int[P];
    int *displs = new int[P];

    // Process (0,0) will have the initial data
    int *x_global = new int[M];
    if (world_rank == 0) {
        for (int i = 0; i < M; i++) {
            x_global[i] = i;
        }
    }

    // Scatter x_global to x_local
    // Calculate the displacement and count arrays for Iscatterv
    for (int i = 0; i < P; i++) {
        sendcounts[i] = x_dist.m(i);
        displs[i] = i * m;
    }

    // Scatter x_global to x_local using Iscatterv
    MPI_Iscatterv(x_global, sendcounts, displs, MPI_INT, x_local, m, MPI_INT, 0, row_comm, &request);

    MPI_Wait(&request, MPI_STATUS_IGNORE);

    delete[] sendcounts;
    delete[] displs;

    // Broadcast x_local to all processes in the row
    MPI_Request bcast_request;
    MPI_Ibcast(x_local, m, MPI_INT, 0, col_comm, &bcast_request);
    MPI_Wait(&bcast_request, MPI_STATUS_IGNORE);

    int nominal1 = M/P; int extra1 = M%P;
    int nominal2 = M/Q; int extra2 = M%Q;

    for(int i = 0; i < m; i++) // m is the local size of the vector x[]
    { 
        // x local to global: given that this element is (p,i), what is its global index I?
        int I = i + ((p < extra1) ? (nominal1+1)*p :
                (extra1*(nominal1+1)+(p-extra1)*nominal1));

        // so to what (qhat,jhat) does this element of the original global vector go?
        int qhat = (I < extra2*(nominal2+1)) ? I/(nominal2+1) : 
                                                (extra2+(I-extra2*(nominal2+1))/nominal2);
        int jhat = I - ((qhat < extra2) ? (nominal2+1)*qhat :
                (extra2*(nominal2+1) + (qhat-extra2)*nominal2));

        if(qhat == q)  // great, this process has an element of y!
        { 
            y_local[jhat] = x_local[i];
        }
    }

    MPI_Request reduce_request;
    MPI_Iallreduce(MPI_IN_PLACE, y_local, n, MPI_INT, MPI_SUM, col_comm, &reduce_request);
    MPI_Wait(&reduce_request, MPI_STATUS_IGNORE);

    //print the results
    std::cout << "World Rank " << world_rank  << "(" << p << ", " << q << ")" << " has the following values:" << std::endl;
    for (int i = 0; i < m; i++) {
        std::cout << " x[" << i << "] = " << x_local[i] << std::endl;
    }
    for (int i = 0; i < m; i++) {
        std::cout << " y[" << i << "] = " << y_local[i] << std::endl;
    }

    delete[] x_local;
    delete[] y_local;
    delete[] x_global;


    MPI_Finalize();
    return 0;
}