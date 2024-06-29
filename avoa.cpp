#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <boost/math/special_functions/gamma.hpp>
#include <mpi.h>
#include "avoa.h"

// Function prototypes
double S1, S2;

// Define your problem-specific parameters here
const int N = 3; // Population size
const int T = 5; // Maximum number of iterations

// Placeholder for Best Vulture categories
double BestVulture1 = 0.0;
double BestVulture2 = 0.0;

double X;
double L1 = 0.8, L2 = 0.2, P1 = 0.5, P2 = 0.5, P3 = 0.5;

double bestFitness = std::numeric_limits<double>::lowest();
double secondBestFitness = std::numeric_limits<double>::lowest();

// Placeholder for PBestVultures
double PBestVulture1 = 0.0;
double PBestVulture2 = 0.0;

// Placeholder for 'Pi' definition
std::vector<double> Pi(N);

// Improved Eq. (1) implementation
double Eq1(double pi, double L1, double L2, double BestVulture1, double BestVulture2)
{
    if (pi >= 0.8) {
        return BestVulture1;
    } else if (pi >= 0.2) {
        return BestVulture2;
    } else {
        return 0.0; // If pi is less than 0.2, return 0.0 or some default value
    }
}

// Corrected Eq. (4) implementation
double Eq4(double rand, double z, int iteration, int maxIterations, double h, double w)
{
    double factor = h * (pow(sin(M_PI / 2.0 * static_cast<double>(iteration) / maxIterations), w) +
                         cos(M_PI / 2.0 * static_cast<double>(iteration) / maxIterations) - 1.0);
    double result = factor + (2.0 * rand + 1.0) * z * (1.0 - static_cast<double>(iteration) / maxIterations);
    
    // Ensure result is non-zero
    if (result == 0.0) {
        result = 0.1;
    }
    
    // std::cout << "Eq4: rand=" << rand << " z=" << z << " iteration=" << iteration << " maxIterations=" << maxIterations << " h=" << h << " w=" << w << " result=" << result << std::endl;
    return result;
}

// Placeholder for Eq. (5) implementation
void Eq5(double &P_i, double R_i, double F, double P1)
{
    double randP1 = static_cast<double>(rand()) / RAND_MAX;
    if (P1 >= randP1)
    {
        Eq6(P_i, R_i, F);
    }
    else
    {
        Eq8(P_i, R_i, F);
    }
}

// Placeholder for Eq. (6) implementation
void Eq6(double &P_i, double R_i, double F)
{
    double X = 2.0 * static_cast<double>(rand()) / RAND_MAX;
    P_i = R_i - fabs(X * R_i - P_i) * F;
}

// Placeholder for Eq. (7) implementation
double Eq7(double R_i, double P_i)
{
    return fabs(2.0 * static_cast<double>(rand()) / RAND_MAX * R_i - P_i);
}

// Placeholder for Eq. (8) implementation
void Eq8(double &P_i, double R_i, double F)
{
    double rand2 = static_cast<double>(rand()) / RAND_MAX;
    double rand3 = static_cast<double>(rand()) / RAND_MAX;
    double lb = 0.1; // Modified to ensure non-zero range
    double ub = 1.0;

    P_i = R_i - F + rand2 * ((ub - lb) * rand3 + lb);
}

// Placeholder for Eq. (10) implementation
void Eq10(double &P_i, double R_i, double F)
{
    double D_i = Eq7(R_i, P_i);
    P_i = D_i * (F + static_cast<double>(rand()) / RAND_MAX) - Eq11(R_i, P_i);
}

// Placeholder for Eq. (11) implementation
double Eq11(double R_i, double P_i)
{
    return R_i - P_i;
}

// Placeholder for Eq. (12) implementation
void Eq12(double &S1, double &S2, double R_i, double P_i)
{
    double rand5 = static_cast<double>(rand()) / RAND_MAX;
    double rand6 = static_cast<double>(rand()) / RAND_MAX;

    S1 = R_i * (rand5 * P_i / (2 * M_PI)) * cos(P_i);
    S2 = R_i * (rand6 * P_i / (2 * M_PI)) * sin(P_i);
}

// Placeholder for Eq. (13) implementation
void Eq13(double &P_i, double R_i)
{
    P_i = R_i - (S1 + S2);
}

// Placeholder for Eq. (15) implementation
void Eq15(double &A1, double &A2, double BestVulture1_i, double BestVulture2_i, double P_i, double F)
{
    A1 = BestVulture1_i - (BestVulture1_i * P_i) / (BestVulture1_i - pow(P_i, 2.0) * F);
    A2 = BestVulture2_i - (BestVulture2_i * P_i) / (BestVulture2_i - pow(P_i, 2.0) * F);
}

// Placeholder for Eq. (16) implementation
void Eq16(double &P_i, double A1, double A2)
{
    P_i = (A1 + A2) / 2.0;
}

// Placeholder for Eq. (17) implementation
void Eq17(double &P_i, double R_i, double F)
{
    double d_t = R_i - P_i;
    double levy = LevyFlight();
    P_i = R_i - fabs(d_t) * F * levy;
}

// Placeholder for LevyFlight implementation
double LevyFlight()
{
    double beta = 1.5; // Placeholder, replace with actual value
    double u = static_cast<double>(rand()) / RAND_MAX;
    double v = static_cast<double>(rand()) / RAND_MAX;

    double sigma = pow((boost::math::tgamma(1.0 + beta) * sin(M_PI * beta / 2.0) /
                        (boost::math::tgamma(1.0 + beta * 2.0) * beta * pow(2.0, (beta - 1) / 2.0))),
                       1 / beta);

    return 0.01 * u * sigma / pow(fabs(v), 1.0 / beta);
}

void AVOA_MPI(int rank, int size)
{
    // Distribute the population among processes
    int local_population_size = N / size;
    std::vector<double> Pi_local(local_population_size);

    // Initialize the random population Pi(i=1,2,...,N)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 1.0); // Modified to avoid zero values
    
    for (int i = 0; i < local_population_size; ++i)
    {
        Pi_local[i] = dis(gen);
        std::cout << "Initialization: Process " << rank << " - Vulture " << i << " - Initial Location: " << Pi_local[i] << std::endl;
    }

    for (int iteration = 0; iteration < T; ++iteration)
    {
        // Calculate fitness values for local population
        double sumFi_local = 0.0;
        for (int i = 0; i < local_population_size; ++i)
        {
            double z = Pi_local[i]; // Use the current position as `z`
            double rand_val = dis(gen);
            double Fi_local = Eq4(rand_val, z, iteration, T, 0.0, 1.0);
            std::cout << "Fitness Calculation: Process " << rank << " - Vulture " << i << " - z: " << z << " - rand_val: " << rand_val << " - Fi_local: " << Fi_local << std::endl;
            sumFi_local += Fi_local;
        }

        // Gather the sum of fitness values from all processes
        double sumFi_global;
        MPI_Allreduce(&sumFi_local, &sumFi_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (sumFi_global == 0) {
            std::cerr << "Error: sumFi_global is zero, cannot proceed with calculation." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Calculate pi values for local population
        for (int i = 0; i < local_population_size; ++i)
        {
            double z = Pi_local[i]; // Use the current position as `z`
            double Fi_local = Eq4(dis(gen), z, iteration, T, 0.0, 1.0);
            double pi = (sumFi_global != 0) ? Fi_local / sumFi_global : 0.0;
            std::cout << "Process " << rank << " - Vulture " << i << " - pi: " << pi << std::endl;

            // Perform Equation 1 using pi
            double R_i = Eq1(pi, L1, L2, BestVulture1, BestVulture2);

            // Update the location Vulture based on conditions
            double F = Eq4(dis(gen), z, iteration, T, 0.0, 1.0);

            // Update Best Vultures based on fitness values
            double currentFitness = Eq1(Pi_local[i], L1, L2, BestVulture1, BestVulture2);
            if (currentFitness > bestFitness)
            {
                secondBestFitness = bestFitness;
                bestFitness = currentFitness;

                PBestVulture2 = PBestVulture1;
                PBestVulture1 = Pi_local[i];
            }
            else if (currentFitness > secondBestFitness)
            {
                secondBestFitness = currentFitness;
                PBestVulture2 = Pi_local[i];
            }

            if (fabs(F) >= 1)
            {
                if (P1 >= dis(gen))
                {
                    Eq6(Pi_local[i], R_i, F);
                }
                else
                {
                    Eq8(Pi_local[i], R_i, F);
                }
            }
            else
            {
                if (fabs(F) >= 0.5)
                {
                    if (P2 >= dis(gen))
                    {
                        Eq10(Pi_local[i], R_i, F);
                    }
                    else
                    {
                        Eq13(Pi_local[i], R_i);
                    }
                }
                else
                {
                    if (P3 >= dis(gen))
                    {
                        Eq16(Pi_local[i], R_i, F);
                    }
                    else
                    {
                        Eq17(Pi_local[i], R_i, F);
                    }
                }
            }

            // Improved debug output for intermediate values
            std::cout << "-----------------------------------------------" << std::endl;
            std::cout << "Process " << rank << " - Iteration " << iteration
                      << " - Vulture " << i << std::endl;
            std::cout << "Location: " << Pi_local[i]
                    //   << " - Fitness Value: " << currentFitness
                      << " - R_i: " << R_i
                      << " - F: " << F << std::endl;
            std::cout << "-----------------------------------------------" << std::endl;
        }

        // Gather the updated population from all processes
        MPI_Gather(Pi_local.data(), local_population_size, MPI_DOUBLE, Pi.data(), local_population_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Stopping condition check

        // Barrier to synchronize processes
        MPI_Barrier(MPI_COMM_WORLD);
    }
}
