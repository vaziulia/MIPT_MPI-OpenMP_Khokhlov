#include <stdio.h>  
#include <stdlib.h>  
#include <mpi.h>  
#include <math.h>  

double f(double x) {  
    return sqrt(4 - x * x);  
}  

int main(int argc, char *argv[]) {  
    int rank, size;  
    double a = 0.0, b = 2.0;  
    long long int N; // Change N to long long int 

    // Initialize MPI 
    MPI_Init(&argc, &argv);  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);  

    // Check for command-line arguments 
    if (argc < 2) {  
        if (rank == 0) {  
            printf("Usage: %s <number_of_intervals>\n", argv[0]);  
        }  
        MPI_Finalize();  
        return 1; // Exit if no argument is provided  
    }  

    // Convert argument to long long int 
    N = atoll(argv[1]); // Use atoll instead of atoi for long long int 
    double h = (b - a) / (double)N; // Interval width 
    double s_, s = 0.0;   
    double start_time, end_time;  

    start_time = MPI_Wtime();  

    // Divide work among processes 
    long long int local_N = N / size; // Number of subintervals for each process  
    double local_a = a + rank * local_N * h; // Start of interval for process  
    double local_b = local_a + local_N * h; // End of interval for process  

    // Calculate integral using the trapezoidal rule  
    s_ = (f(local_a) + f(local_b)) / 2.0;  

    long long int i; // Declare i as long long int 
    for (i = 1; i < local_N; i++) {  
        double x = local_a + i * h;  
        s_ += f(x);  
    }  
    s_ *= h;  

    if (rank == 0) {  
        // Receive results from other processes  
        for (i = 1; i < size; i++) {  
            double temp_s;  
            MPI_Recv(&temp_s, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  
            s += temp_s;  
        }  
        s += s_; // Add result from the zero process  

        // Print final sum and rank of the processor
        printf("Final Integral= %.6f from processor with rank %d\n", s, rank);  
    } else {  
        // Send results to the zero process  
        MPI_Send(&s_, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);  

        // Optionally print local sum from other processes
        //printf("Processor with rank %d computed partial sum= %.6f\n", rank, s_);  
    }  
    end_time = MPI_Wtime(); 
    if (rank == 0) {
        printf("Time %f sec\n", end_time - start_time);
	FILE *f = fopen("list2.txt", "a");
	fprintf(f, "Processes: %d, Time: %f seconds, Integral: %f\n", size, end_time - start_time, s);
	fclose(f);
    }
 

    // Finalize MPI 
    MPI_Finalize();  

    return 0; 
}
