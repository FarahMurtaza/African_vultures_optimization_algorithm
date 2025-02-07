#ifndef AVOA_H
#define AVOA_H

#include <vector>

double LevyFlight(int iteration, int maxIterations, double lowerBound, double upperBound, double beta = 1.5);
void Eq6(double& P_i, double R_i, double F);
void Eq8(double& P_i, double R_i, double F);
double Eq4(double rand, double z, int iteration, int maxIterations, double h, double w);
double Eq11(double R_i, double P_i);
void Eq13(double& P_i, double R_i);
void Eq10(double& P_i, double R_i, double F);
void Eq16(double& P_i, double A1, double A2);
void Eq17(double &P_i, double R_i, double F, int iteration, int maxIterations, double lowerBound, double upperBound);
void AVOA_MPI(int rank, int size, int functionChoice, int populationSize, int maxIterations, double L1, double L2, double w, double P1, double P2, double P3, int dimension);
void AVOA_Optimization(int rank, int size, int functionChoice, int populationSize, int maxIterations, double L1, double L2, double w, double P1, double P2, double P3);



#endif // AVOA_H
