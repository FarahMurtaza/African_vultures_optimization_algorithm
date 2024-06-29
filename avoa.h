#ifndef AVOA_H
#define AVOA_H

#include <vector>

double LevyFlight();
void Eq6(double& P_i, double R_i, double F);
void Eq8(double& P_i, double R_i, double F);
double Eq11(double R_i, double P_i);
void Eq13(double& P_i, double R_i);
void Eq10(double& P_i, double R_i, double F);
void Eq16(double& P_i, double A1, double A2);
void Eq17(double& P_i, double R_i, double F);
void AVOA_MPI(int rank, int size);

#endif // AVOA_H
