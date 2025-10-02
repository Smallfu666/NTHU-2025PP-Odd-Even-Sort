#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <boost/sort/spreadsort/spreadsort.hpp>

void merge_and_swap(std::vector<float> &data, const std::vector<float> &recv_buf, int recv_n, bool keep_smaller, std::vector<float> &result)
{
    if (data.empty() || recv_n == 0)
        return;

    if (keep_smaller && data.back() <= recv_buf[0])
        return;
    if (!keep_smaller && data[0] >= recv_buf[recv_n - 1])
        return;

    const int m = data.size();

    if (keep_smaller)
    {
        int i = 0, j = 0, k = 0;
        while (k < m)
        {
            if (j >= recv_n || data[i] <= recv_buf[j])
            {
                result[k++] = data[i++];
            }
            else
            {
                result[k++] = recv_buf[j++];
            }
        }
    }
    else
    {
        int i = m - 1, j = recv_n - 1, k = m - 1;
        while (k >= 0)
        {
            if (j < 0 || (i >= 0 && data[i] >= recv_buf[j]))
            {
                result[k--] = data[i--];
            }
            else
            {
                result[k--] = recv_buf[j--];
            }
        }
    }

    data.swap(result);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    long long local_n, offset;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4)
    {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " N input_file output_file [--profile]\n";
        MPI_Finalize();
        return 1;
    }

    long long N = atoll(argv[1]);
    std::string input_path = argv[2];
    std::string output_path = argv[3];

    if (size > N && N > 0)
    {
        MPI_Comm_split(MPI_COMM_WORLD, rank < N ? 0 : MPI_UNDEFINED, rank, &comm);
        if (comm == MPI_COMM_NULL)
        {
            MPI_Finalize();
            return 0;
        }
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    long long base_n = 0, rem_n = 0;
    if (N > 0)
    {
        base_n = N / size;
        rem_n = N % size;
    }

    if (rank < rem_n)
    {
        local_n = base_n + 1;
    }
    else
    {
        local_n = base_n;
    }

    if (rank < rem_n)
    {
        offset = rank * (base_n + 1);
    }
    else
    {
        offset = rem_n * (base_n + 1) + (rank - rem_n) * base_n;
    }

    int prev_n = 0;
    if (rank > 0 && N > 0)
    {
        if (rank - 1 < rem_n)
        {
            prev_n = base_n + 1;
        }
        else
        {
            prev_n = base_n;
        }
    }

    int next_n = 0;
    if (rank < size - 1 && N > 0)
    {
        if (rank + 1 < rem_n)
        {
            next_n = base_n + 1;
        }
        else
        {
            next_n = base_n;
        }
    }

    std::vector<float> data;
    if (local_n > 0)
    {
        float *temp_data = (float *)malloc(local_n * sizeof(float));

        MPI_File in_file;
        MPI_File_open(comm, input_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file);
        MPI_File_read_at(in_file, offset * sizeof(float), temp_data, local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
        MPI_File_close(&in_file);

        data.assign(temp_data, temp_data + local_n);

        free(temp_data);
    }

    std::vector<float> recv_buf;
    std::vector<float> merge_buf(local_n);

    int max_neighbor_n;
    if (prev_n > next_n)
        max_neighbor_n = prev_n;
    else
        max_neighbor_n = next_n;
    if (max_neighbor_n > 0)
        recv_buf.resize(max_neighbor_n);

    if (local_n > 0)
    {
        boost::sort::spreadsort::spreadsort(data.begin(), data.end());
    }

    while (true)
    {
        int swapped_this_iteration = 0;

        int partner = (rank % 2 == 0) ? (rank + 1) : (rank - 1);
        if (partner >= 0 && partner < size)
        {
            int partner_n = (rank % 2 == 0) ? next_n : prev_n;

            float my_first = 0, my_last = 0, partner_first = 0, partner_last = 0;
            if (local_n > 0)
            {
                my_first = data.front();
                my_last = data.back();
            }

            MPI_Sendrecv(&my_first, 1, MPI_FLOAT, partner, 0, &partner_first, 1, MPI_FLOAT, partner, 0, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&my_last, 1, MPI_FLOAT, partner, 1, &partner_last, 1, MPI_FLOAT, partner, 1, comm, MPI_STATUS_IGNORE);

            bool needs_merge = false;
            if (local_n > 0 && partner_n > 0)
            {
                if ((rank < partner && my_last > partner_first) || (rank > partner && my_first < partner_last))
                {
                    needs_merge = true;
                }
            }
            if (needs_merge)
            {
                MPI_Sendrecv(data.data(), local_n, MPI_FLOAT, partner, 2, recv_buf.data(), partner_n, MPI_FLOAT, partner, 2, comm, MPI_STATUS_IGNORE);
                merge_and_swap(data, recv_buf, partner_n, rank < partner, merge_buf);
                swapped_this_iteration = 1;
            }
        }

        partner = (rank % 2 != 0) ? (rank + 1) : (rank - 1);
        if (partner >= 0 && partner < size)
        {
            int partner_n = (rank % 2 != 0) ? next_n : prev_n;

            float my_first = 0, my_last = 0, partner_first = 0, partner_last = 0;
            if (local_n > 0)
            {
                my_first = data.front();
                my_last = data.back();
            }

            MPI_Sendrecv(&my_first, 1, MPI_FLOAT, partner, 3, &partner_first, 1, MPI_FLOAT, partner, 3, comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&my_last, 1, MPI_FLOAT, partner, 4, &partner_last, 1, MPI_FLOAT, partner, 4, comm, MPI_STATUS_IGNORE);

            bool needs_merge = false;
            if (local_n > 0 && partner_n > 0)
            {
                if ((rank < partner && my_last > partner_first) || (rank > partner && my_first < partner_last))
                {
                    needs_merge = true;
                }
            }
            if (needs_merge)
            {
                MPI_Sendrecv(data.data(), local_n, MPI_FLOAT, partner, 5, recv_buf.data(), partner_n, MPI_FLOAT, partner, 5, comm, MPI_STATUS_IGNORE);
                merge_and_swap(data, recv_buf, partner_n, rank < partner, merge_buf);
                swapped_this_iteration = 1;
            }
        }

        int total_swapped = 0;
        MPI_Allreduce(&swapped_this_iteration, &total_swapped, 1, MPI_INT, MPI_LOR, comm);
        if (total_swapped == 0)
        {
            break;
        }
    }

    MPI_File out_file;
    MPI_File_open(comm, output_path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out_file);
    if (local_n > 0)
    {
        MPI_File_write_at(out_file, offset * sizeof(float), data.data(), local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&out_file);

    MPI_Finalize();
    return 0;
}