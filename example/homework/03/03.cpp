/*Problem 3 (10pts/8pts):
Modify Problem #1 above, but store y in a scatter distribution (aka wrap-mapped) distribution. For this
case, from global coeffient J on Q processes, then local index is j = J/Q, q = J mod Q, and the number of
elements per process is the same as in the linear load-balanced distribution would produce with N elements
over Q partitions.
*/

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
    int M = 10;

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
    LinearDistribution dist(P, M);

    // Allocate the vectors x and y
    int *x = new int[M];
    int *y = new int[M];
    for (int i = 0; i < M; i++) {
        x[i] = 0;
        y[i] = 0;
    }

    // Process (p,q) will have m elements of x
    m = dist.m(p);
    n = dist.m(q);

    // Process (0,0) will have the initial data
    if (world_rank == 0) {
        for (int i = 0; i < M; i++) {
            x[i] = i;
        }
    }

    // Distribute x down the first column
    MPI_Request request;
    for (int i = 0; i < P; i++) {
        if (p == i) {
            MPI_Ibcast(x, M, MPI_INT, 0, row_comm, &request);
        }
    }
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    // Broadcast x horizontally in each process row
    for (int i = 0; i < Q; i++) {
        if (q == i) {
            MPI_Ibcast(x, M, MPI_INT, 0, col_comm, &request);
        }
    }
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    // Copy x to y
    MPI_Iallreduce(x, y, M, MPI_INT, MPI_SUM, row_comm, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    // Print the result
    std::cout << "Rank " << world_rank << std::endl;
    for (int i = 0; i < M; i++) {
        std::cout << "y[" << i << "] = " << y[i] << std::endl;
    }

    delete[] x;
    delete[] y;


    MPI_Finalize();
    return 0;
}