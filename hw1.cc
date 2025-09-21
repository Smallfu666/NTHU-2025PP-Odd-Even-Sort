#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <mpi.h>

using namespace std;

void oddEvenSort(vector<float> &arr)
{
    const int n = arr.size();
    if (n <= 1)
        return; // 邊界條件優化

    int totalSwaps;
    do
    {
        totalSwaps = 0;

        // Even phase - 優化的交換方式
        for (int i = 0; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                // 手動交換，避免 std::swap 的額外開銷
                float temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                totalSwaps++;
            }
        }

        // Odd phase - 優化的交換方式
        for (int i = 1; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                float temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
                totalSwaps++;
            }
        }
    } while (totalSwaps > 0);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4)
    {
        if (rank == 0)
        {
            cerr << "Usage: " << argv[0] << " N input_file output_file" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    long long N = atoll(argv[1]);
    string input_file = argv[2];
    string output_file = argv[3];

    vector<float> arr(N);

    // Use MPI-IO to read input file (collective operations)
    MPI_File infile;
    MPI_File_open(MPI_COMM_WORLD, input_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);
    if (rank == 0)
    {
        MPI_File_read(infile, arr.data(), N, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&infile);

    // Do the sorting (only rank 0)
    if (rank == 0)
    {
        oddEvenSort(arr);
    }

    // Write output file (collective operations)
    MPI_File outfile;
    MPI_File_open(MPI_COMM_WORLD, output_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile);
    if (rank == 0)
    {
        MPI_File_write(outfile, arr.data(), N, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&outfile);

    MPI_Finalize();
    return 0;
}
