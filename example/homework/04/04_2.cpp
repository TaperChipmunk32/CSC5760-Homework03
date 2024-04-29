/*Problem 4 (28pts):
2. (8pts) Modify the exercise with one stored in linear load-balanced distribution, and the other in the
scatter distribution.
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
    int row_color = world_rank / Q;
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, world_rank, &row_comm);

    // Second split based on ranks mod Q
    int col_color = world_rank % Q;
    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col_color, world_rank, &col_comm);

    int p = world_rank / Q;
    int q = world_rank % Q;
    int m, n;
    LinearDistribution x_dist(P, M);

    // Process (p,q) will have m elements of x
    m = x_dist.m(p);

    // Allocate the vectors x and y
    int *x_local = new int[m];
    int *y_local = new int[m];
    for (int i = 0; i < m; i++) {
        x_local[i] = 0;
        y_local[i] = 0;
    }

    // Process (0,0) will have the initial data
    int *x_global = new int[M];
    if (world_rank == 0) {
        for (int i = 0; i < M; i++) {
            x_global[i] = i;
        }
    }

    MPI_Request request;
    // Scatter x_global to x_local
    // Calculate the displacement and count arrays for Iscatterv
    int *sendcounts = new int[P];
    int *displs = new int[P];
    for (int i = 0; i < P; i++) {
        sendcounts[i] = x_dist.m(i);
        displs[i] = i * m;
    }

    // Scatter x_global to x_local using Iscatterv
    MPI_Iscatterv(x_global, sendcounts, displs, MPI_INT, x_local, m, MPI_INT, 0, row_comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    delete[] sendcounts;
    delete[] displs;

    int nominal1 = M/P; int extra1 = M%P;

    for(int i = 0; i < m; i++) // m is the local size of the vector x[]
    { 
            // x local to global: given that this element is (p,i), what is its global index I?
        int I = i + ((p < extra1) ? (nominal1+1)*p :
                (extra1*(nominal1+1)+(p-extra1)*nominal1));

        // so to what (qhat,jhat) does this element of the original global vector go?
        int qhat = I%Q;
        int jhat = I/Q;

        if(qhat == q)  // great, this process has an element of y!
        { 
            y_local[jhat] = x_local[i];
        }
    }

    //print the results
    std::cout << "Rank " << world_rank << " has the following values:" << std::endl;
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