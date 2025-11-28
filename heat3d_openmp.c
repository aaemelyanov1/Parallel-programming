#include "common.h"
#include <omp.h>

double*** allocate_grid() {
    double ***grid = malloc(NX * sizeof(double**));
    for (int i = 0; i < NX; i++) {
        grid[i] = malloc(NY * sizeof(double*));
        for (int j = 0; j < NY; j++) grid[i][j] = malloc(NZ * sizeof(double));
    }
    return grid;
}

void free_grid(double ***grid) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) free(grid[i][j]);
        free(grid[i]);
    }
    free(grid);
}

// Сохранение результатов
void save_results(double ***temp, const char* filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing\n");
        return;
    }
    
    // Сохраняем 3D сетку в бинарный файл
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fwrite(temp[i][j], sizeof(double), NZ, file);
        }
    }
    
    fclose(file);
    printf("Results saved to %s\n", filename);
}

int main(int argc, char *argv[]) {
    int num_threads = atoi(argv[1]);
    double ***temp = allocate_grid();
    double ***new_temp = allocate_grid();
    double dx2=DX*DX, dy2=DY*DY, dz2=DZ*DZ, factor=DT*ALPHA;
    
    // Инициализация
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int i = 0; i < NX; i++)
        for (int j = 0; j < NY; j++)
            for (int k = 0; k < NZ; k++)
                temp[i][j][k] = (abs(i-NX/2)<CUBE_SIZE && abs(j-NY/2)<CUBE_SIZE && abs(k-NZ/2)<CUBE_SIZE) ? INITIAL_TEMP : BOUNDARY_TEMP;
    
    double start = omp_get_wtime();
    
    for (int step = 0; step < NUM_STEPS; step++) {
        #pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 1; i < NX-1; i++) {
            for (int j = 1; j < NY-1; j++) {
                for (int k = 1; k < NZ-1; k++) {
                    double d2x = (temp[i+1][j][k] - 2*temp[i][j][k] + temp[i-1][j][k]) / dx2;
                    double d2y = (temp[i][j+1][k] - 2*temp[i][j][k] + temp[i][j-1][k]) / dy2;
                    double d2z = (temp[i][j][k+1] - 2*temp[i][j][k] + temp[i][j][k-1]) / dz2;
                    new_temp[i][j][k] = temp[i][j][k] + factor * (d2x + d2y + d2z);
                }
            }
        }
        
        #pragma omp parallel for collapse(2) num_threads(num_threads)
        for (int i = 1; i < NX-1; i++)
            for (int j = 1; j < NY-1; j++)
                for (int k = 1; k < NZ-1; k++)
                    temp[i][j][k] = new_temp[i][j][k];
    }
    
    printf("OpenMP (%d threads): %.2f sec\n", num_threads, omp_get_wtime()-start);
    save_results(temp, "openmp_results.bin");
    free_grid(temp); free_grid(new_temp);
    return 0;
}