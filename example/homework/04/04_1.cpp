/*Problem 4 (28pts):
1. (8pts) Using the results of Problem #1 and #3 to compute the dot product of two distinct vectors of
length M , one stored initially horizontally in linear load-balanced distribution, and one stored initially
vertically in linear load-balanced distribution. The scalar result should be in all processes at the end
of the computation.
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
    int M = 12;

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
    
    int *x_sendcounts = new int[P];
    int *x_displs = new int[P];
    int *y_sendcounts = new int[Q];
    int *y_displs = new int[Q];

    // Process (0,0) will have the initial data
    int *x_global = new int[M];
    int *y_global = new int[M];
    if (world_rank == 0) {
        for (int i = 0; i < M; i++) {
            x_global[i] = i;
            y_global[i] = i;
        }
    }

    // Scatter global to local
    // Calculate the displacement and count arrays for Iscatterv
    for (int i = 0; i < P; i++) {
        x_sendcounts[i] = x_dist.m(i);
        x_displs[i] = i * m;
    }
    for (int i = 0; i < Q; i++) {
        y_sendcounts[i] = y_dist.m(i);
        y_displs[i] = i * n;
    }

    MPI_Request scatter_requests[2];

    // Scatter global to local using Iscatterv
    MPI_Iscatterv(x_global, x_sendcounts, x_displs, MPI_INT, x_local, m, MPI_INT, 0, row_comm, &scatter_requests[0]);
    MPI_Iscatterv(y_global, y_sendcounts, y_displs, MPI_INT, y_local, n, MPI_INT, 0, col_comm, &scatter_requests[1]);
    
    MPI_Waitall(2, scatter_requests, MPI_STATUSES_IGNORE);

    delete[] x_sendcounts;
    delete[] x_displs;
    delete[] y_sendcounts;
    delete[] y_displs;

    MPI_Request bcast_requests[2];
    
    // Broadcast x_local to all processes in the row
    MPI_Ibcast(x_local, m, MPI_INT, 0, col_comm, &bcast_requests[0]);
    
    // Broadcast y_local to all processes in the column
    MPI_Ibcast(y_local, n, MPI_INT, 0, row_comm, &bcast_requests[1]);
    
    MPI_Waitall(2, bcast_requests, MPI_STATUSES_IGNORE);

    // Compute the dot product
    int dot_product = 0;
    for (int i = 0; i < m; i++) {
        dot_product += x_local[i] * y_local[i];
    }

    // Reduce the dot products to the root process
    int global_dot_product;
    MPI_Request reduce_request;
    MPI_Ireduce(&dot_product, &global_dot_product, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &reduce_request);
    MPI_Wait(&reduce_request, MPI_STATUS_IGNORE);

    // Broadcast result to all processes
    MPI_Request bcast_request;
    MPI_Ibcast(&global_dot_product, 1, MPI_INT, 0, MPI_COMM_WORLD, &bcast_request);
    MPI_Wait(&bcast_request, MPI_STATUS_IGNORE);

    if (world_rank == 0) {
        std::cout << "The dot product of the two vectors is " << global_dot_product << std::endl;
    }

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
    delete[] y_global;


    MPI_Finalize();
    return 0;
}