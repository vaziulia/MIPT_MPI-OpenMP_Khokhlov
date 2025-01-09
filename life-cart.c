/*
 * Author: Nikolay Khokhlov <k_h@inbox.ru>, 2016
 */

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <mpi.h>

#define ind(i, j) (((i + l->nx) % l->nx) + ((j + l->ny) % l->ny) * (l->nx))

typedef struct {
	int nx, ny;
	int *u0;
	int *u1;
	int steps;
	int save_steps;

	/* MPI */
	int start_x, stop_x, start_y, stop_y;
	int rank;
	int size;
	MPI_Comm Cart_comm, Comm_row, Comm_col;
	int dim_size[2];
	int coords[2];
    
	// для передачи данных между процессами для расчета
	MPI_Datatype exchange_col_t;
	MPI_Datatype exchange_row_t;

	// для передачи данных на последний процессор для сохранения поля
	MPI_Datatype collect_row_t;
	MPI_Datatype collect_col_t;

} life_t;

void life_init(const char *path, life_t *l);
void life_free(life_t *l);
void life_step(life_t *l);
void life_save_vtk(const char *path, life_t *l);
void life_collect(life_t *l);
void data_exchange(life_t *l);
void decompisition(const int nx, const int ny, int *dim_size, int *coords, int *start_x, int *stop_x, int *start_y, int *stop_y);
void get_size(const int n, int dim_size, const int coord, int *start, int *stop);

int main(int argc, char **argv)
{
	double t1, t2;
    int rank, size;
	MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if (size == 1) {
        FILE *fptr;
        fptr = fopen("mpi-runtime-cart.txt", "w");
        fprintf(fptr, "number_process   time,s\n");
        fclose(fptr);
    }
    // старт замера времени
    if (rank == size - 1) {
        t1 = MPI_Wtime();
    }

	if (argc != 2) {
		printf("Usage: %s input file.\n", argv[0]);
		return 0;
	}
	life_t l;
	life_init(argv[1], &l);

	int i;
	char buf[100];
	for (i = 0; i < l.steps; i++) {
/*
		if (i % l.save_steps == 0) {
			life_collect(&l);
			if (l.rank == l.size - 1) {
				sprintf(buf, "vtk/life_cart_%06d.vtk", i);
				printf("Saving step %d to '%s'.\n", i, buf);
				life_save_vtk(buf, &l);
			}
		}
*/

		life_step(&l);
        data_exchange(&l);
	}

	life_free(&l);



	// конец замера времени
    if (rank == size - 1) {
        t2 = MPI_Wtime();
        double dt = t2 - t1;  // секунды
    
        // запись в файл
        FILE *fptr;
        fptr = fopen("mpi-runtime-cart.txt", "a");
        fprintf(fptr, "%d, %f\n", l.size, dt);
        fclose(fptr);
    }


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


	// создание декартовой топологии
	int periods[2];
	periods[0] = 1;
	periods[1] = 1;
	l->dim_size[0] = 0;
	l->dim_size[1] = 0;
	MPI_Dims_create(l->size, 2, l->dim_size);
	MPI_Cart_create(MPI_COMM_WORLD, 2, l->dim_size, periods, 1, &(l->Cart_comm));

	// координаты процесса
	MPI_Cart_coords(l->Cart_comm, l->rank, 2, l->coords);

	// декомпозиция
	decompisition(l->nx, l->ny, l->dim_size, l->coords, &(l->start_x), &(l->stop_x), &(l->start_y), &(l->stop_y));

	// вспомогательные коммуникаторы
	MPI_Comm_split(MPI_COMM_WORLD, l->coords[0], l->coords[1], &(l->Comm_col));
	MPI_Comm_split(MPI_COMM_WORLD, l->coords[1], l->coords[0], &(l->Comm_row));

	// задание типов данных для передачи сообщений между процессами
	// как поколоночная декомпозиция
	MPI_Type_vector(l->stop_y - l->start_y, 1, l->nx, MPI_INT, &(l->exchange_row_t));
    MPI_Type_commit(&(l->exchange_row_t));
	// как построковая декомпозиция
	MPI_Type_vector(1, l->stop_x - l->start_x, l->nx, MPI_INT, &(l->exchange_col_t));
    MPI_Type_commit(&(l->exchange_col_t));

	// задание типов данных для сбора данных на одном процессе
	// collect_row_t - тип данных, описывающий поле любого процесса, кроме мб последнего в этой строке
	// высота поля (l->stop_y - l->start_y) одинаковая для всех процессов в данной строке
	// ширина поля (l->stop_x - l->start_x) одинакова для всех процессов, кроме мб последнего столбца. Поэтому берем ширину из нулевого ранга, т.к. он гарантированно в нижнем левом углу
	int start_x_0, stop_x_0, start_y_0, stop_y_0;
	int coords_0[2];
	coords_0[0] = 0;
	coords_0[1] = 0;
	decompisition(l->nx, l->ny, l->dim_size, coords_0, &start_x_0, &stop_x_0, &start_y_0, &stop_y_0);
	
	MPI_Type_vector(l->stop_y - l->start_y, stop_x_0 - start_x_0, l->nx, MPI_INT, &(l->collect_row_t));
    MPI_Type_commit(&(l->collect_row_t));

	// collect_col_t - тип данных, описывающий поле любого процесса, кроме мб последнего в этом столбце
	// передуем всю имеющуюся строку с в последний в столбце процесс
	// высота строки такая же, как высота поля нулевого процесса
	MPI_Type_vector(stop_y_0 - start_y_0, l->nx, l->nx, MPI_INT, &(l->collect_col_t));
    MPI_Type_commit(&(l->collect_col_t));
}

void life_free(life_t *l)
{
	free(l->u0);
	free(l->u1);
	l->nx = l->ny = 0;
	MPI_Type_free(&(l->exchange_row_t));
	MPI_Type_free(&(l->exchange_col_t));
	MPI_Type_free(&(l->collect_row_t));
	MPI_Type_free(&(l->collect_col_t));
	MPI_Comm_free(&(l->Cart_comm));
	MPI_Comm_free(&(l->Comm_row));
	MPI_Comm_free(&(l->Comm_col));
}

void life_save_vtk(const char *path, life_t *l)
{
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

void life_step(life_t *l)
{
	int i, j;
	for (j = l->start_y; j < l->stop_y; j++) {
		for (i = l->start_x; i < l->stop_x; i++) {
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

void life_collect(life_t *l)
{
	// передача в рамках строки (поколоночная декомпозиция)
	// передаем свое поле последнему процессу в строке
	int size_row, rank_row;
	MPI_Comm_size(l->Comm_row, &size_row);
	MPI_Comm_rank(l->Comm_row, &rank_row);
	if (rank_row == size_row - 1) {
		int i;
		for(i = 0; i < size_row - 1; i++) {
			// есть i-й процесс в Comm_row. В рамках строки все процессы имеют одинаковый start_y
			// Нужно получить start_x i-го процесса в строке. Его координата по x равна i
			// делаем декомпозицию с правильным coords_i[0] = i, coords_i[1] = 0 - нам не важно, т.к. не будем использовать.
			int start_x_i, stop_x_i, start_y_i, stop_y_i;
			int coords_i[2];
			coords_i[0] = i;
			coords_i[1] = 0;
			decompisition(l->nx, l->ny, l->dim_size, coords_i, &start_x_i, &stop_x_i, &start_y_i, &stop_y_i);
			MPI_Recv(l->u0 + ind(start_x_i, l->start_y), 1, 
					l->collect_row_t, i, 0, l->Comm_row, MPI_STATUS_IGNORE);
		}
	} else {
		MPI_Send(l->u0 + ind(l->start_x, l->start_y), 1, 
			l->collect_row_t, size_row - 1, 0, l->Comm_row);
	}

	// передача в рамках последней колонки
	if (l->coords[0] == l->dim_size[0] - 1) {
		int size_col, rank_col;
		MPI_Comm_size(l->Comm_col, &size_col);
		MPI_Comm_rank(l->Comm_col, &rank_col);
		if (rank_col == size_col - 1) {
			int i;
			for(i = 0; i < size_col - 1; i++) {
				// есть i-й процесс в Comm_col
				// Нужно получить start_y i-го процесса. Его координата по y равна i
				// делаем декомпозицию с правильным coords_i[1] = i, coords_i[0] = 0 - нам не важно, т.к. не будем использовать.
				int start_x_i, stop_x_i, start_y_i, stop_y_i;
				int coords_i[2];
				coords_i[0] = 0;
				coords_i[1] = i;
				decompisition(l->nx, l->ny, l->dim_size, coords_i, &start_x_i, &stop_x_i, &start_y_i, &stop_y_i);
				MPI_Recv(l->u0 + ind(0, start_y_i), 1, 
						l->collect_col_t, i, 0, l->Comm_col, MPI_STATUS_IGNORE);
			}
		} else {
			MPI_Send(l->u0 + ind(0, l->start_y), 1, 
				l->collect_col_t, size_col - 1, 0, l->Comm_col);
		}
	}
}

// обмен данными
void data_exchange(life_t *l)
{
	// передача в рамках строки (поколоночная декомпозиция)
	int size_row, rank_row;
	MPI_Comm_size(l->Comm_row, &size_row);
	MPI_Comm_rank(l->Comm_row, &rank_row);
    if (size_row > 1){
        if (rank_row == 0) {
            MPI_Send(l->u0 + ind(l->stop_x-1, l->start_y), 1, l->exchange_row_t, 1, 0, l->Comm_row);
            MPI_Recv(l->u0 + ind(l->start_x-1, l->start_y), 1, l->exchange_row_t, size_row-1, 0, l->Comm_row, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(l->u0 + ind(l->start_x-1, l->start_y), 1, l->exchange_row_t, rank_row-1, 0, l->Comm_row, MPI_STATUS_IGNORE);
            MPI_Send(l->u0 + ind(l->stop_x-1, l->start_y), 1, l->exchange_row_t, (rank_row + size_row + 1) % (size_row), 0, l->Comm_row);
        }

        if (rank_row == 0) {
            MPI_Send(l->u0 + ind(l->start_x, l->start_y), 1, l->exchange_row_t, size_row-1, 0, l->Comm_row);
            MPI_Recv(l->u0 + ind(l->stop_x, l->start_y), 1, l->exchange_row_t, 1, 0, l->Comm_row, MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(l->u0 + ind(l->stop_x, l->start_y), 1, l->exchange_row_t, (rank_row + size_row + 1) % (size_row), 0, l->Comm_row, MPI_STATUS_IGNORE);
            MPI_Send(l->u0 + ind(l->start_x, l->start_y), 1, l->exchange_row_t, rank_row-1, 0, l->Comm_row);
        }
    }

	// передача в рамках столбца (построчная декомпозиция)
	int size_col, rank_col;
	MPI_Comm_size(l->Comm_col, &size_col);
	MPI_Comm_rank(l->Comm_col, &rank_col);
	if (size_col > 1){
        if (rank_col == 0) {
            MPI_Send(l->u0 + ind(l->start_x,   l->stop_y -1), 1, l->exchange_col_t, 1, 0, l->Comm_col);
			MPI_Send(l->u0 + ind(l->start_x-1, l->stop_y -1), 1, MPI_INT, 1, 0, l->Comm_col); // верхний левый угол
			MPI_Send(l->u0 + ind(l->stop_x,    l->stop_y -1), 1, MPI_INT, 1, 0, l->Comm_col); // верхний правый угол
            MPI_Recv(l->u0 + ind(l->start_x,   l->start_y-1), 1, l->exchange_col_t, size_col-1, 0, l->Comm_col, MPI_STATUS_IGNORE);
			MPI_Recv(l->u0 + ind(l->start_x-1, l->start_y-1), 1, MPI_INT, size_col-1, 0, l->Comm_col, MPI_STATUS_IGNORE); // нижний левый угол
			MPI_Recv(l->u0 + ind(l->stop_x,    l->start_y-1), 1, MPI_INT, size_col-1, 0, l->Comm_col, MPI_STATUS_IGNORE); // нижний правый угол
        } else {
            MPI_Recv(l->u0 + ind(l->start_x,   l->start_y-1), 1, l->exchange_col_t, rank_col-1, 0, l->Comm_col, MPI_STATUS_IGNORE);
			MPI_Recv(l->u0 + ind(l->start_x-1, l->start_y-1), 1, MPI_INT, rank_col-1, 0, l->Comm_col, MPI_STATUS_IGNORE); // нижний левый угол
			MPI_Recv(l->u0 + ind(l->stop_x,    l->start_y-1), 1, MPI_INT, rank_col-1, 0, l->Comm_col, MPI_STATUS_IGNORE); // нижний правый угол
            MPI_Send(l->u0 + ind(l->start_x,   l->stop_y -1), 1, l->exchange_col_t, (rank_col + size_col + 1) % (size_col), 0, l->Comm_col);
			MPI_Send(l->u0 + ind(l->start_x-1, l->stop_y -1), 1, MPI_INT, (rank_col + size_col + 1) % (size_col), 0, l->Comm_col); // верхний левый угол
			MPI_Send(l->u0 + ind(l->stop_x,    l->stop_y -1), 1, MPI_INT, (rank_col + size_col + 1) % (size_col), 0, l->Comm_col); // верхний правый угол
        }

        if (rank_col == 0) {
            MPI_Send(l->u0 + ind(l->start_x,   l->start_y), 1, l->exchange_col_t, size_col-1, 0, l->Comm_col);
			MPI_Send(l->u0 + ind(l->start_x-1, l->start_y), 1, MPI_INT, size_col-1, 0, l->Comm_col); // нижний левый угол
			MPI_Send(l->u0 + ind(l->stop_x,    l->start_y), 1, MPI_INT, size_col-1, 0, l->Comm_col); // нижний правый угол
            MPI_Recv(l->u0 + ind(l->start_x,   l->stop_y ), 1, l->exchange_col_t, 1, 0, l->Comm_col, MPI_STATUS_IGNORE);
			MPI_Recv(l->u0 + ind(l->start_x-1, l->stop_y ), 1, MPI_INT, 1, 0, l->Comm_col, MPI_STATUS_IGNORE); // верхний левый угол
			MPI_Recv(l->u0 + ind(l->stop_x,    l->stop_y ), 1, MPI_INT, 1, 0, l->Comm_col, MPI_STATUS_IGNORE); // верхний правый угол
        } else {
            MPI_Recv(l->u0 + ind(l->start_x,   l->stop_y ), 1, l->exchange_col_t, (rank_col + size_col + 1) % (size_col), 0, l->Comm_col, MPI_STATUS_IGNORE);
			MPI_Recv(l->u0 + ind(l->start_x-1, l->stop_y ), 1, MPI_INT, (rank_col + size_col + 1) % (size_col), 0, l->Comm_col, MPI_STATUS_IGNORE); // верхний левый угол
			MPI_Recv(l->u0 + ind(l->stop_x,    l->stop_y ), 1, MPI_INT, (rank_col + size_col + 1) % (size_col), 0, l->Comm_col, MPI_STATUS_IGNORE); // верхний правый угол
            MPI_Send(l->u0 + ind(l->start_x,   l->start_y), 1, l->exchange_col_t, rank_col-1, 0, l->Comm_col);
			MPI_Send(l->u0 + ind(l->start_x-1, l->start_y), 1, MPI_INT, rank_col-1, 0, l->Comm_col); // нижний левый угол
			MPI_Send(l->u0 + ind(l->stop_x,    l->start_y), 1, MPI_INT, rank_col-1, 0, l->Comm_col); // нижний правый угол
        }
    }
}

void decompisition(const int nx, const int ny, int *dim_size, int *coords, int *start_x, int *stop_x, int *start_y, int *stop_y)
{
	// определение размеров поля процесса по оси x
	get_size(nx, dim_size[0], coords[0], start_x, stop_x);

	// определение размеров поля процесса по оси y
	get_size(ny, dim_size[1], coords[1], start_y, stop_y);
}

void get_size(const int n, int dim_size, const int coord, int *start, int *stop)
{
	// определение размеров поля процесса
	int l = n / dim_size; // длина куска
	*start = l * coord;
	*stop = *start + l;
	// последний процесс
	if (coord == dim_size - 1) *stop = n;
}
