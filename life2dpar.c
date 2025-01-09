#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define ind(i, j) (((i + l->nx) % l->nx) + ((j + l->ny) % l->ny) * (l->nx))

typedef struct {
	int nx, ny;
	int *u0;
	int *u1;
	int steps;
	int save_steps;

	/* MPI */
	int start, stop;
	int rank;
	int size;
} life_t;

void life_init(const char *path, life_t *l);
void life_free(life_t *l);
void life_step(life_t *l);
void life_save_vtk(const char *path, life_t *l);
void decompisition(const int n, const int p, const int k, int *start, int *stop);
void data_exchange(life_t *l);

int main(int argc, char **argv)
{
    clock_t start, end;
    
	MPI_Init(&argc, &argv);
	if (argc != 2) {
		printf("Usage: %s input file.\n", argv[0]);
		return 0;
	}
	life_t l;
	life_init(argv[1], &l);

    // старт замера времени
    if (l.rank == 0) {
        start = clock();
    }
	
	int i;
	char buf[100];
	for (i = 0; i < l.steps; i++) {
        // сохраняет последний процесс
		if (i % l.save_steps == 0) {
            if (l.rank == l.size - 1) {
                sprintf(buf, "life_par_%06d.vtk", i);
                printf("Saving step %d to '%s'.\n", i, buf);
            }
            life_save_vtk(buf, &l);
		}
		life_step(&l);
        data_exchange(&l);
	}

    // конец замера времени
    if (l.rank == 0) {
        end = clock();
        double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
    
        // запись в файл
        FILE *fptr;
        fptr = fopen("mpi-runtime.txt", "a");
        fprintf(fptr, "%d, %f\n", l.size, time_taken);
        fclose(fptr);
    }

    life_free(&l);
	MPI_Finalize();
	return 0;
}

/**
 * Загрузить входную конфигурацию.
 * Формат файла, число шагов, как часто сохранять, размер поля, затем идут координаты заполненых клеток:
 * steps
 * save_steps
 * nx ny
 * i1 j2
 * i2 j2
 */
void life_init(const char *path, life_t *l)
{
	FILE *fd = fopen(path, "r");
	assert(fd);
	assert(fscanf(fd, "%d\n", &l->steps));
	assert(fscanf(fd, "%d\n", &l->save_steps));
	printf("Steps %d, save every %d step.\n", l->steps, l->save_steps);
	assert(fscanf(fd, "%d %d\n", &l->nx, &l->ny));
	printf("Field size: %dx%d\n", l->nx, l->ny);

	l->u0 = (int*)calloc(l->nx * l->ny, sizeof(int));
	l->u1 = (int*)calloc(l->nx * l->ny, sizeof(int));
	
	int i, j, r, cnt;
	cnt = 0;
	while ((r = fscanf(fd, "%d %d\n", &i, &j)) != EOF) {
		l->u0[ind(i, j)] = 1;
		cnt++;
	}
	printf("Loaded %d life cells.\n", cnt);
	fclose(fd);


	/* MPI */
	MPI_Comm_size(MPI_COMM_WORLD, &(l->size));
	MPI_Comm_rank(MPI_COMM_WORLD, &(l->rank));
	decompisition(l->ny, l->size, l->rank, &(l->start), &(l->stop));
}

void life_free(life_t *l)
{
	free(l->u0);
	free(l->u1);
	l->nx = l->ny = 0;
}

void life_save_vtk(const char *path, life_t *l)
{
    // // чтобы работало на одном потоке
    // if (l->size > 1) {
        // все передают свои данные на последний процесс, он все принимает, затем сохраняет все поле целиком
        // все процессы кроме последнего имеют одинаковый размер поля
        if (l->rank == l->size - 1) {
            // recv from all
            int i;
            for (i = 0; i < l->size - 1; ++i) {
                int l_i = l->ny / l->size; // длинна куска по y от процессора i
                int start_i = l_i * i;  // start i-го процессора
                MPI_Recv(l->u0 + ind(0, start_i), l->nx * l_i, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            //send to the last process
            MPI_Send(l->u0 + ind(0, l->start), (l->nx) * (l->stop - l->start), MPI_INT, l->size - 1, 1, MPI_COMM_WORLD);
        }
    // }

    // запись в файл
    if (l->rank == l->size - 1) {
        FILE *f;
        int i1, i2, j;
        f = fopen(path, "w");
        assert(f);
        fprintf(f, "# vtk DataFile Version 3.0\n");
        fprintf(f, "Created by write_to_vtk2d\n");
        fprintf(f, "ASCII\n");
        fprintf(f, "DATASET STRUCTURED_POINTS\n");
        fprintf(f, "DIMENSIONS %d %d 1\n", l->nx+1, l->ny+1);
        fprintf(f, "SPACING %d %d 0.0\n", 1, 1);
        fprintf(f, "ORIGIN %d %d 0.0\n", 0, 0);
        fprintf(f, "CELL_DATA %d\n", l->nx * l->ny);
        
        fprintf(f, "SCALARS life int 1\n");
        fprintf(f, "LOOKUP_TABLE life_table\n");
        for (i2 = 0; i2 < l->ny; i2++) {
            for (i1 = 0; i1 < l->nx; i1++) {
                fprintf(f, "%d\n", l->u0[ind(i1, i2)]);
            }
        }
        fclose(f);
    }
}

void life_step(life_t *l)
{
	int i, j;
	for (j = l->start; j < l->stop; j++) {
		for (i = 0; i < l->nx; i++) {
			int n = 0;
			n += l->u0[ind(i+1, j)];
			n += l->u0[ind(i+1, j+1)];
			n += l->u0[ind(i,   j+1)];
			n += l->u0[ind(i-1, j)];
			n += l->u0[ind(i-1, j-1)];
			n += l->u0[ind(i,   j-1)];
			n += l->u0[ind(i-1, j+1)];
			n += l->u0[ind(i+1, j-1)];
			l->u1[ind(i,j)] = 0;
			if (n == 3 && l->u0[ind(i,j)] == 0) {
				l->u1[ind(i,j)] = 1;
			}
			if ((n == 3 || n == 2) && l->u0[ind(i,j)] == 1) {
				l->u1[ind(i,j)] = 1;
			}
		}
	}
	int *tmp;
	tmp = l->u0;
	l->u0 = l->u1;
	l->u1 = tmp;
}

// обмен данными
void data_exchange(life_t *l)
{
    // // чтобы работало на одном потоке
    // if (l->size > 1){
        if (l->rank == 0) {
            MPI_Send(l->u0 + ind(0, l->stop-1), l->nx, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(l->u0 + ind(0, l->start-1), l->nx, MPI_INT, l->size-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(l->u0 + ind(0, l->start-1), l->nx, MPI_INT, l->rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(l->u0 + ind(0, l->stop-1), l->nx, MPI_INT, (l->rank + l->size + 1) % (l->size), 0, MPI_COMM_WORLD);
        }

        if (l->rank == 0) {
            MPI_Send(l->u0 + ind(0, l->start), l->nx, MPI_INT, l->size-1, 0, MPI_COMM_WORLD);
            MPI_Recv(l->u0 + ind(0, l->stop), l->nx, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(l->u0 + ind(0, l->stop), l->nx, MPI_INT, (l->rank + l->size + 1) % (l->size), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(l->u0 + ind(0, l->start), l->nx, MPI_INT, l->rank-1, 0, MPI_COMM_WORLD);
        }
    // }
}

void decompisition(const int n, const int p, const int k, int *start, int *stop)
{
	int l = n / p; // длинна куска
	*start = l * k;
	*stop = *start + l;
	if (k == n - 1) *stop = n;
}
