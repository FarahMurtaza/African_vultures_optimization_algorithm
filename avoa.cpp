#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <mpi.h>
#include <boost/math/special_functions/gamma.hpp>
#include <cstdlib>

static double LOWER_BOUND = 0.0;
static double UPPER_BOUND = 0.0;

// Global best values fitness + solution vector
static double bestFitness = std::numeric_limits<double>::infinity();
static double secondBestFitness = std::numeric_limits<double>::infinity();
static std::vector<double> PBestVulture1; // best position
static std::vector<double> PBestVulture2; // second best position

// Benchmark Functions
double F1(const std::vector<double> &x)
{
    double sum = 0.0;
    for (double xi : x)
        sum += xi * xi;
    return sum;
}

double F2(const std::vector<double> &x)
{
    double sum = 0.0;
    double product = 1.0;
    for (double xi : x)
    {
        sum += std::fabs(xi);
        product *= std::fabs(xi);
    }
    return sum + product;
}

double F3(const std::vector<double> &x)
{
    double total = 0.0;
    for (size_t i = 0; i < x.size(); i++)
    {
        double partialSum = 0.0;
        for (size_t j = 0; j <= i; j++)
        {
            partialSum += x[j];
        }
        total += (partialSum * partialSum);
    }
    return total;
}

using FitnessFunction = double (*)(const std::vector<double> &);

FitnessFunction selectFunction(int choice)
{
    switch (choice)
    {
    case 1:
        return F1;
    case 2:
        return F2;
    case 3:
        return F3;
    default:
        return F1;
    }
}

// Random helpers
static thread_local std::mt19937 rng(std::random_device{}());
static double rand01()
{
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

// Eq1
void Eq1(const std::vector<double> &best1,
         const std::vector<double> &best2,
         int iteration, int maxIterations,
         double L1, double L2,
         std::vector<double> &R)
{
    double alpha = L2 + (L1 - L2) * (1.0 - (double)iteration / (double)maxIterations);
    size_t dim = best1.size();
    R.resize(dim);
    for (size_t d = 0; d < dim; d++)
    {
        R[d] = alpha * best1[d] + (1.0 - alpha) * best2[d];
    }
}

// Eq4
double Eq4(double randVal, double z, int iteration, int maxIterations, double h, double w)
{
    double t = (double)iteration;
    double T = (double)maxIterations;

    double factor = h * (std::pow(std::sin(M_PI / 2.0 * t / T), w) + std::cos(M_PI / 2.0 * t / T) - 1.0);
    double F = factor + (2.0 * randVal + 1.0) * z * (1.0 - t / T);
    return F;
}

void Eq6(std::vector<double> &P, const std::vector<double> &R, double F)
{
    for (size_t d = 0; d < P.size(); d++)
    {
        double randX = rand01();
        P[d] = R[d] - std::fabs(randX * R[d] - P[d]) * F;
    }
}

void Eq8(std::vector<double> &P, const std::vector<double> &R, double F)
{
    double r2 = rand01();
    double r3 = rand01();
    double lb = 0.1, ub = 1.0;
    double step = (ub - lb) * r3 + lb;

    for (size_t d = 0; d < P.size(); d++)
    {
        P[d] = R[d] - F + r2 * step;
    }
}

void Eq10(std::vector<double> &P, const std::vector<double> &R, double F)
{
    for (size_t d = 0; d < P.size(); d++)
    {
        double rr = rand01();
        double D_d = std::fabs(rr * R[d] - P[d]);
        double rr2 = rand01();
        P[d] = D_d * (F + rr2) - (R[d] - P[d]);
    }
}

// void Eq13(std::vector<double> &P, const std::vector<double> &R) {
//     for (size_t d = 0; d < P.size(); d++) {
//         double S1 = R[d] * std::cos(P[d]) * 0.1;
//         double S2 = R[d] * std::sin(P[d]) * 0.1;
//         P[d] = R[d] - (S1 + S2);
//     }
// }

void Eq13(std::vector<double> &P, const std::vector<double> &R)
{
    for (size_t d = 0; d < P.size(); d++)
    {
        double rand5 = rand01();
        double rand6 = rand01();
        double factor1 = rand5 * (P[d] / (2.0 * M_PI));
        double factor2 = rand6 * (P[d] / (2.0 * M_PI));
        double S1 = R[d] * factor1 * std::cos(P[d]);
        double S2 = R[d] * factor2 * std::sin(P[d]);
        P[d] = R[d] - (S1 + S2);
    }
}

void Eq16(std::vector<double> &P, const std::vector<double> &BestVulture1, const std::vector<double> &BestVulture2, double F)
{
    // for (size_t d = 0; d < P.size(); d++) {
    //     P[d] = 0.5*(A1[d] + A2[d]);
    // }

    // size_t dim = P.size();
    // if (BestVulture1.size() != dim || BestVulture2.size() != dim) {
    //     throw std::invalid_argument("Vector dimensions must match.");
    // }

    for (size_t d = 0; d < P.size(); d++)
    {
        // Calculate A1 for dimension d:
        // Avoid division by zero by checking the denominator.
        double denom1 = BestVulture1[d] - (P[d] * P[d] * F);
        double A1;
        if (std::fabs(denom1) < 1e-10)
        {
            // If the denominator is nearly zero, use BestVulture1[d] as a fallback.
            A1 = BestVulture1[d];
        }
        else
        {
            A1 = BestVulture1[d] - (BestVulture1[d] * P[d]) / denom1;
        }

        // Calculate A2 for dimension d:
        double denom2 = BestVulture2[d] - (P[d] * P[d] * F);
        double A2;
        if (std::fabs(denom2) < 1e-10)
        {
            A2 = BestVulture2[d];
        }
        else
        {
            A2 = BestVulture2[d] - (BestVulture2[d] * P[d]) / denom2;
        }

        // Aggregate A1 and A2 to update the position P:
        P[d] = 0.5 * (A1 + A2);
    }
}

// Levy flight
double LevyFlight()
{
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    double u = dist01(rng);
    double v = dist01(rng);
    double beta = 1.5, beta2 = 1.5;

    double numerator = boost::math::tgamma(1.0 + beta) * std::sin(M_PI * beta / 2.0);
    double denominator = boost::math::tgamma((1.0 + beta2)) * beta * (2.0 * (beta - 1.0) / 2.0);
    double sigma = std::pow(numerator / denominator, 1.0 / beta);

    double step = sigma / std::pow(std::fabs(v), 1.0 / beta);
    return 0.01 * step * u;
}

void Eq17(std::vector<double> &P, const std::vector<double> &R, double F,
          int iteration, int maxIterations)
{
    for (size_t d = 0; d < P.size(); d++)
    {
        double d_t = R[d] - P[d];
        double levy = LevyFlight();
        P[d] = R[d] - std::fabs(d_t) * F * levy;
    }
}

// The main AVOA function
void AVOA_MPI(int rank, int size,
              int functionChoice,
              int populationSize,
              int maxIterations,
              double L1, double L2,
              double w,
              double P1, double P2, double P3,
              int dimension)
{

    if (functionChoice == 2)
    {
        // F2 range is [-10, 10]
        LOWER_BOUND = -10.0;
        UPPER_BOUND = 10.0;
    }
    else
    {
        // F1, F3 => [-100, 100]
        LOWER_BOUND = -100.0;
        UPPER_BOUND = 100.0;
    }

    // Split the population across ranks
    int localPopSize = populationSize / size;
    if (populationSize % size != 0 && rank == 0)
    {
        std::cerr << "Warning: populationSize not divisible by #ranks.\n";
    }

    // localPopulation: each rank only sees localPopSize vultures
    std::vector<std::vector<double>> localPopulation(localPopSize, std::vector<double>(dimension));

    // random init
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distPos(LOWER_BOUND, UPPER_BOUND);

    for (int i = 0; i < localPopSize; i++)
    {
        for (int d = 0; d < dimension; d++)
        {
            localPopulation[i][d] = distPos(gen);
        }
    }

    FitnessFunction fitnessFunc = selectFunction(functionChoice);

    bestFitness = std::numeric_limits<double>::infinity();
    secondBestFitness = std::numeric_limits<double>::infinity();

    PBestVulture1.resize(dimension, 0.0);
    PBestVulture2.resize(dimension, 0.0);

    std::vector<double> iterationBestLog(maxIterations, 0.0);

    double hValue = 1.0;

    // 4) Main iteration loop
    for (int iter = 0; iter < maxIterations; iter++)
    {

        // (A) Evaluate local pop => find local best & 2nd best
        double localBestFit = std::numeric_limits<double>::infinity();
        std::vector<double> localBestPos(dimension);

        double localSecondBestFit = std::numeric_limits<double>::infinity();
        std::vector<double> localSecondBestPos(dimension);

        for (int i = 0; i < localPopSize; i++)
        {
            double fVal = fitnessFunc(localPopulation[i]);
            if (fVal < localBestFit)
            {
                localSecondBestFit = localBestFit;
                localSecondBestPos = localBestPos;

                localBestFit = fVal;
                localBestPos = localPopulation[i];
            }
            else if (fVal < localSecondBestFit)
            {
                localSecondBestFit = fVal;
                localSecondBestPos = localPopulation[i];
            }
        }

        // (B) MPI_Allreduce to find global min
        double globalBestFit = 0.0;
        MPI_Allreduce(&localBestFit, &globalBestFit, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

        bool iAmOwner = (std::fabs(localBestFit - globalBestFit) < 1e-14);
        if (iAmOwner)
        {
            bestFitness = localBestFit;
            secondBestFitness = localSecondBestFit;
            PBestVulture1 = localBestPos;
            PBestVulture2 = localSecondBestPos;
        }

        // broadcast bestFitness, secondBestFitness, best positions
        MPI_Bcast(&bestFitness, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&secondBestFitness, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank != 0)
        {
            PBestVulture1.resize(dimension);
            PBestVulture2.resize(dimension);
        }
        MPI_Bcast(PBestVulture1.data(), dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(PBestVulture2.data(), dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::cout << "Iteration " << iter
                      << " | bestFitness=" << bestFitness
                      << std::endl;
        }
        iterationBestLog[iter] = bestFitness;

        // (C) Update each local vulture (each individual)
        for (int i = 0; i < localPopSize; i++)
        {
            // Create a reference vector R by Eq1 using the global best positions.
            std::vector<double> R(dimension);
            Eq1(PBestVulture1, PBestVulture2, iter, maxIterations, L1, L2, R);

            // Compute a representative F value for the individual.
            // For example, average the Eq4 value over all dimensions.
            double F_val = 0.0;
            for (int d = 0; d < dimension; d++)
            {
                F_val += Eq4(rand01(), localPopulation[i][d], iter, maxIterations, hValue, w);
            }
            F_val /= dimension;
            double absF = std::fabs(F_val);

            // Select one of the update operators based on absF and the probabilities.
            if (absF >= 1.0)
            {
                if (rand01() < P1)
                {
                    Eq6(localPopulation[i], R, F_val);
                }
                else
                {
                    Eq8(localPopulation[i], R, F_val);
                }
            }
            else if (absF >= 0.5)
            {
                if (rand01() < P2)
                {
                    Eq10(localPopulation[i], R, F_val);
                }
                else
                {
                    Eq13(localPopulation[i], R);
                }
            }
            else
            {
                if (rand01() < P3)
                {
                    Eq16(localPopulation[i], PBestVulture1, PBestVulture2, F_val);
                }
                else
                {
                    Eq17(localPopulation[i], R, F_val, iter, maxIterations);
                }
            }

            // Clamp each dimension of the updated individual to the defined bounds.
            for (int d = 0; d < dimension; d++)
            {
                if (localPopulation[i][d] < LOWER_BOUND)
                    localPopulation[i][d] = LOWER_BOUND;
                if (localPopulation[i][d] > UPPER_BOUND)
                    localPopulation[i][d] = UPPER_BOUND;
            }
        }

    } // end localPop loop
    // end iteration loop

    // rank=0 prints final result, writes fitness_log
    if (rank == 0)
    {
        std::cout << "\n=== AVOA finished ===\n";
        std::cout << "Final best fitness=" << bestFitness << "\n";
        std::cout << "Best position=[ ";
        for (double c : PBestVulture1)
            std::cout << c << " ";
        std::cout << "]\n";

        std::ofstream outFile("fitness_log.txt");
        for (double val : iterationBestLog)
        {
            outFile << val << "\n";
        }
        outFile.close();

        std::system("python3 analyze_convergence.py");
    }
}
