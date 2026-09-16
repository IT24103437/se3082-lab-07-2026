#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: the number of processes must evenly divide %d.\n",
                    N);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int chunk_size = N / size;
    int *array = NULL;

    // Only the root process stores and initializes the complete array.
    if (rank == 0) {
        array = malloc((size_t)N * sizeof(*array));
        if (array == NULL) {
            fprintf(stderr, "Root could not allocate the complete array.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        for (int i = 0; i < N; ++i) {
            array[i] = i + 1;
        }
        printf("Root filled array with values 1 to %d\n", N);
    }

    // Every process stores only the chunk that it needs to sum.
    int *local_chunk = malloc((size_t)chunk_size * sizeof(*local_chunk));
    if (local_chunk == NULL) {
        fprintf(stderr, "Rank %d: unable to allocate its local chunk.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    double start = MPI_Wtime();

    MPI_Scatter(array, chunk_size, MPI_INT,
                local_chunk, chunk_size, MPI_INT,
                0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < chunk_size; ++i) {
        local_sum += local_chunk[i];
    }

    int start_idx = rank * chunk_size;
    int end_idx = start_idx + chunk_size;
    printf("  Rank %d: summed indices [%d, %d) => local_sum = %lld\n",
           rank, start_idx, end_idx, local_sum);

    // Keep the point-to-point result collection from Exercise 1.
    if (rank != 0) {
        MPI_Send(&local_sum, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    } else {
        long long total_sum = local_sum;

        for (int source = 1; source < size; ++source) {
            long long received_sum;
            MPI_Recv(&received_sum, 1, MPI_LONG_LONG, source, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += received_sum;
        }

        double elapsed = MPI_Wtime() - start;
        long long expected = (long long)N * (N + 1) / 2;

        printf("\n[Scatter] Total sum = %lld\n", total_sum);
        printf("[Scatter] Expected  = %lld\n", expected);
        printf("[Scatter] Correct?  = %s\n",
               total_sum == expected ? "YES" : "NO");
        printf("[Scatter] Time      = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    free(array);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
