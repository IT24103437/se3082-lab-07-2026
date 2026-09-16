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

    // Each rank receives the sum of all local sums up to its own rank.
    long long prefix_sum = 0;
    MPI_Scan(&local_sum, &prefix_sum, 1, MPI_LONG_LONG,
             MPI_SUM, MPI_COMM_WORLD);

    long long sum_before_me = prefix_sum - local_sum;
    long long last_value = (long long)(rank + 1) * chunk_size;
    long long expected_prefix = last_value * (last_value + 1) / 2;

    printf("Rank %d: local_sum = %lld, prefix_sum = %lld, "
           "sum_before_me = %lld, correct_prefix = %s\n",
           rank, local_sum, prefix_sum, sum_before_me,
           prefix_sum == expected_prefix ? "YES" : "NO");

    // The final rank's prefix is the sum of the entire array.
    if (rank == size - 1) {
        double elapsed = MPI_Wtime() - start;
        long long expected_total = (long long)N * (N + 1) / 2;

        printf("\n[Scan] Final prefix = %lld\n", prefix_sum);
        printf("[Scan] Expected     = %lld\n", expected_total);
        printf("[Scan] Correct?     = %s\n",
               prefix_sum == expected_total ? "YES" : "NO");
        printf("[Scan] Time         = %.4f sec\n", elapsed);
    }

    free(local_chunk);
    free(array);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
