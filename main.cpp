#include "avoa.h"
#include <mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Gather parameters from user on rank=0, broadcast
    double L1, L2, w, P1, P2, P3;
    int populationSize, maxIterations, functionChoice, dimension;

    if (rank == 0)
    {
        std::cout << "Enter L1: ";
        std::cin >> L1;
        std::cout << "Enter L2: ";
        std::cin >> L2;
        std::cout << "Enter w : ";
        std::cin >> w;
        std::cout << "Enter P1: ";
        std::cin >> P1;
        std::cout << "Enter P2: ";
        std::cin >> P2;
        std::cout << "Enter P3: ";
        std::cin >> P3;
        std::cout << "Enter populationSize (e.g. 30, 50, 100,...): ";
        std::cin >> populationSize;
        std::cout << "Enter maxIterations: ";
        std::cin >> maxIterations;
        std::cout << "Select function (1=F1, 2=F2, 3=F3,...): ";
        std::cin >> functionChoice;
        std::cout << "Enter dimension (e.g. 30, 100, 500, 1000): ";
        std::cin >> dimension;
    }

    // Broadcast them
    MPI_Bcast(&L1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&w, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P3, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&populationSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxIterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&functionChoice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dimension, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Now call your AVOA function
    AVOA_MPI(rank, size, functionChoice,
             populationSize, maxIterations,
             L1, L2, w, P1, P2, P3,
             dimension);

    MPI_Finalize();
    return 0;
}
