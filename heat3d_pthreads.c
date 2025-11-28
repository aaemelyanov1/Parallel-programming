#include "common.h"
#include <pthread.h>

typedef struct {
    double ***temp;
    double ***new_temp;
    int start_x, end_x;
    int start_y, end_y;
    pthread_barrier_t *barrier;
} ThreadData;

double*** allocate_grid() {
    double ***grid = malloc(NX * sizeof(double**));
    for (int i = 0; i < NX; i++) {
        grid[i] = malloc(NY * sizeof(double*));
        for (int j = 0; j < NY; j++) {
            grid[i][j] = malloc(NZ * sizeof(double));
        }
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

void initialize(double ***temp, double value) {
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                temp[i][j][k] = value;
            }
        }
    }
}

void set_hot_cube(double ***temp) {
    int center_x = NX/2, center_y = NY/2, center_z = NZ/2;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (abs(i - center_x) < CUBE_SIZE && 
                    abs(j - center_y) < CUBE_SIZE && 
                    abs(k - center_z) < CUBE_SIZE) {
                    temp[i][j][k] = INITIAL_TEMP;
                }
            }
        }
    }
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

void* solve_slice(void *arg) {
    ThreadData *data = (ThreadData*)arg;
    double dx2=DX*DX, dy2=DY*DY, dz2=DZ*DZ, factor=DT*ALPHA;
    
    for (int step = 0; step < NUM_STEPS; step++) {
        // Вычисления только для внутренних точек в назначенной области
        for (int i = data->start_x; i < data->end_x; i++) {
            if (i < 1 || i >= NX-1) continue;
            
            for (int j = data->start_y; j < data->end_y; j++) {
                if (j < 1 || j >= NY-1) continue;
                
                for (int k = 1; k < NZ-1; k++) {
                    double d2x = (data->temp[i+1][j][k] - 2*data->temp[i][j][k] + data->temp[i-1][j][k]) / dx2;
                    double d2y = (data->temp[i][j+1][k] - 2*data->temp[i][j][k] + data->temp[i][j-1][k]) / dy2;
                    double d2z = (data->temp[i][j][k+1] - 2*data->temp[i][j][k] + data->temp[i][j][k-1]) / dz2;
                    data->new_temp[i][j][k] = data->temp[i][j][k] + factor * (d2x + d2y + d2z);
                }
            }
        }
        
        pthread_barrier_wait(data->barrier);
        
        // Копирование обновленных значений в назначенной области
        for (int i = data->start_x; i < data->end_x; i++) {
            if (i < 1 || i >= NX-1) continue;
            
            for (int j = data->start_y; j < data->end_y; j++) {
                if (j < 1 || j >= NY-1) continue;
                
                for (int k = 1; k < NZ-1; k++) {
                    data->temp[i][j][k] = data->new_temp[i][j][k];
                }
            }
        }
        
        pthread_barrier_wait(data->barrier);
        
        // Устанавливаем граничные условия для назначенной области
        for (int i = data->start_x; i < data->end_x; i++) {
            for (int j = data->start_y; j < data->end_y; j++) {
                for (int k = 0; k < NZ; k++) {
                    // Границы по X (только если наша область содержит граничные точки)
                    if (i == 0 || i == NX-1) {
                        data->temp[i][j][k] = BOUNDARY_TEMP;
                        data->new_temp[i][j][k] = BOUNDARY_TEMP;
                    }
                    // Границы по Y (только если наша область содержит граничные точки)
                    if (j == 0 || j == NY-1) {
                        data->temp[i][j][k] = BOUNDARY_TEMP;
                        data->new_temp[i][j][k] = BOUNDARY_TEMP;
                    }
                    // Границы по Z (все потоки обрабатывают свои части Z-границ)
                    if (k == 0 || k == NZ-1) {
                        data->temp[i][j][k] = BOUNDARY_TEMP;
                        data->new_temp[i][j][k] = BOUNDARY_TEMP;
                    }
                }
            }
        }
        
        pthread_barrier_wait(data->barrier);
    }
    return NULL;
}

int main(int argc, char *argv[]) {
    int num_threads = atoi(argv[1]);
    double ***temp = allocate_grid();
    double ***new_temp = allocate_grid();
    
    // Инициализируем все точки boundary temperature (20.0)
    initialize(temp, BOUNDARY_TEMP);
    initialize(new_temp, BOUNDARY_TEMP);
    
    // Устанавливаем горячий куб
    set_hot_cube(temp);
    set_hot_cube(new_temp);
    
    pthread_t threads[num_threads];
    ThreadData data[num_threads];
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, num_threads);
    
    // Определяем размеры блоков для распределения по X и Y
    int threads_x, threads_y;
    
    // Находим оптимальное распределение потоков по осям X и Y
    if (num_threads <= 4) {
        threads_x = num_threads;
        threads_y = 1;
    } else {
        // Для большего числа потоков распределяем более равномерно
        threads_x = (int)sqrt(num_threads);
        threads_y = num_threads / threads_x;
        // Корректируем, если произведение не равно num_threads
        while (threads_x * threads_y < num_threads) {
            threads_y++;
        }
    }
    
    int x_per_thread = NX / threads_x;
    int y_per_thread = NY / threads_y;
    
    clock_t start = clock();
    
    // Создаем потоки с распределением по X и Y
    int thread_idx = 0;
    for (int tx = 0; tx < threads_x && thread_idx < num_threads; tx++) {
        for (int ty = 0; ty < threads_y && thread_idx < num_threads; ty++) {
            data[thread_idx].temp = temp; 
            data[thread_idx].new_temp = new_temp; 
            data[thread_idx].barrier = &barrier;
            
            data[thread_idx].start_x = tx * x_per_thread;
            data[thread_idx].end_x = (tx == threads_x-1) ? NX : (tx+1) * x_per_thread;
            
            data[thread_idx].start_y = ty * y_per_thread;
            data[thread_idx].end_y = (ty == threads_y-1) ? NY : (ty+1) * y_per_thread;
            
            pthread_create(&threads[thread_idx], NULL, solve_slice, &data[thread_idx]);
            thread_idx++;
        }
    }
    
    for (int t = 0; t < num_threads; t++) pthread_join(threads[t], NULL);
    
    printf("Pthreads (%d threads): %.2f sec\n", num_threads, ((double)(clock()-start))/CLOCKS_PER_SEC);
    
    // Проверяем минимальную температуру перед сохранением
    double min_temp = temp[0][0][0];
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (temp[i][j][k] < min_temp) min_temp = temp[i][j][k];
            }
        }
    }
    
    save_results(temp, "pthreads_results.bin");
    
    pthread_barrier_destroy(&barrier);
    free_grid(temp); 
    free_grid(new_temp);
    return 0;
}