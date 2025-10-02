#include <mpi.h>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <boost/sort/spreadsort/spreadsort.hpp>

// 用於兩階段通訊的邊界值 struct
struct Boundary
{
    float first;
    float last;
};

// MPI process info
struct MPI_Ctx
{
    int rank = 0, size = 1;
    long long local_n = 0;
    long long offset = 0;
    MPI_Comm comm = MPI_COMM_WORLD;
};

// 用於計時的 struct
struct Timer
{
    double total = 0.0, io_read = 0.0, io_write = 0.0;
    double local_sort = 0.0, merge = 0.0;
    double comm_probe = 0.0, comm_data = 0.0; // 將通訊時間細分
};

void merge_and_swap(std::vector<float> &data, const std::vector<float> &recv_buf, int recv_n, bool keep_smaller)
{
    std::vector<float> result;
    result.reserve(data.size());

    if (keep_smaller)
    {
        auto it1 = data.cbegin();
        auto it2 = recv_buf.cbegin();
        auto it2_end = recv_buf.cbegin() + recv_n;
        while (result.size() < data.size())
        {
            if (it1 != data.cend() && (it2 == it2_end || *it1 <= *it2))
            {
                result.push_back(*it1++);
            }
            else if (it2 != it2_end)
            {
                result.push_back(*it2++);
            }
            else
            {
                break;
            }
        }
    }
    else
    {
        auto it1 = data.crbegin();
        auto it2 = recv_buf.crbegin() + (recv_buf.size() - recv_n);
        auto it2_end = recv_buf.crend();
        result.resize(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (it1 != data.crend() && (it2 == it2_end || *it1 >= *it2))
            {
                result[data.size() - 1 - i] = *it1++;
            }
            else if (it2 != it2_end)
            {
                result[data.size() - 1 - i] = *it2++;
            }
            else
            {
                break;
            }
        }
    }
    data.swap(result);
}

/*inline void merge_and_swap(std::vector<float>& data,
                           const float* __restrict recv_buf, int recv_n,
                           bool keep_smaller,
                           float* __restrict tmp) {
    const int m = static_cast<int>(data.size());

    if (keep_smaller) {
        // 從小到大，取前 m 個
        int i = 0;         // data index
        int j = 0;         // recv index
        int k = 0;         // tmp index
        while (k < m) {
            if (j >= recv_n || (i < m && data[i] <= recv_buf[j])) {
                tmp[k++] = data[i++];
            } else {
                tmp[k++] = recv_buf[j++];
            }
        }
    } else {
        // 從大到小，取後 m 個（反向合併）
        int i = m - 1;           // data index from end
        int j = recv_n - 1;      // recv index from end
        int k = m - 1;           // tmp index from end
        while (k >= 0) {
            if (j < 0 || (i >= 0 && data[i] >= recv_buf[j])) {
                tmp[k--] = data[i--];
            } else {
                tmp[k--] = recv_buf[j--];
            }
        }
    }

    // 回寫（memcpy 最快，且不拋例外）
    std::memcpy(data.data(), tmp, sizeof(float) * m);
}*/

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Ctx ctx;
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

    if (argc < 4)
    {
        if (ctx.rank == 0)
            std::cerr << "Usage: " << argv[0] << " N input_file output_file [--profile]\n";
        MPI_Finalize();
        return 1;
    }
    bool do_profile = (argc == 5 && strcmp(argv[4], "--profile") == 0);

    double t_start = MPI_Wtime();
    Timer timer;
    double t0;

    long long N = atoll(argv[1]);
    std::string input_path = argv[2];
    std::string output_path = argv[3];

    if (ctx.size > N && N > 0)
    {
        MPI_Comm_split(MPI_COMM_WORLD, ctx.rank < N ? 0 : MPI_UNDEFINED, ctx.rank, &ctx.comm);
        if (ctx.comm == MPI_COMM_NULL)
        {
            MPI_Finalize();
            return 0;
        }
        MPI_Comm_rank(ctx.comm, &ctx.rank);
        MPI_Comm_size(ctx.comm, &ctx.size);
    }
    // 分配每個 process 的資料數量與偏移量
    long long base_n = (N > 0) ? N / ctx.size : 0;
    long long rem_n = (N > 0) ? N % ctx.size : 0;
    ctx.local_n = base_n + (ctx.rank < rem_n ? 1 : 0);

    if (ctx.rank < rem_n)
    {
        ctx.offset = ctx.rank * (base_n + 1);
    }
    else
    {
        ctx.offset = rem_n * (base_n + 1) + (ctx.rank - rem_n) * base_n;
    }

    int prev_n = (ctx.rank > 0 && N > 0) ? (base_n + (ctx.rank - 1 < rem_n ? 1 : 0)) : 0;
    int next_n = (ctx.rank < ctx.size - 1 && N > 0) ? (base_n + (ctx.rank + 1 < rem_n ? 1 : 0)) : 0;
    // buffer for local data and communication
    std::vector<float> data(ctx.local_n);
    std::vector<float> recv_buf;
    std::vector<float> tmp_buf(ctx.local_n); //

    int max_neighbor_n;
    if (prev_n > next_n)
        max_neighbor_n = prev_n;
    else
        max_neighbor_n = next_n;
    if (max_neighbor_n > 0)
        recv_buf.resize(max_neighbor_n);

    t0 = MPI_Wtime();
    MPI_File in_file;
    MPI_File_open(ctx.comm, input_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file);
    if (ctx.local_n > 0)
    {
        MPI_File_read_at(in_file, ctx.offset * sizeof(float), data.data(), ctx.local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&in_file);
    timer.io_read = MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    if (ctx.local_n > 0)
    {
        // std::sort(data.begin(), data.end());
        boost::sort::spreadsort::spreadsort(data.begin(), data.end());
    }
    timer.local_sort = MPI_Wtime() - t0;

    double comm_probe_acc = 0.0, comm_data_acc = 0.0, merge_acc = 0.0;
    while (true)
    {
        int swapped_this_iteration = 0;

        // --- Even Phase ---
        int partner = (ctx.rank % 2 == 0) ? (ctx.rank + 1) : (ctx.rank - 1);
        if (partner >= 0 && partner < ctx.size)
        {
            int partner_n = (ctx.rank % 2 == 0) ? next_n : prev_n;

            Boundary my_b = {}, partner_b = {};
            if (ctx.local_n > 0)
                my_b = {data.front(), data.back()};
            t0 = MPI_Wtime();
            MPI_Sendrecv(&my_b, 2, MPI_FLOAT, partner, 0, &partner_b, 2, MPI_FLOAT, partner, 0, ctx.comm, MPI_STATUS_IGNORE);
            comm_probe_acc += MPI_Wtime() - t0;

            bool needs_merge = false;
            if (ctx.local_n > 0 && partner_n > 0)
            {
                if ((ctx.rank < partner && my_b.last > partner_b.first) || (ctx.rank > partner && my_b.first < partner_b.last))
                {
                    needs_merge = true;
                }
            }
            if (needs_merge)
            {
                t0 = MPI_Wtime();
                MPI_Sendrecv(data.data(), ctx.local_n, MPI_FLOAT, partner, 1, recv_buf.data(), partner_n, MPI_FLOAT, partner, 1, ctx.comm, MPI_STATUS_IGNORE);
                comm_data_acc += MPI_Wtime() - t0;

                t0 = MPI_Wtime();
                merge_and_swap(data, recv_buf, partner_n, ctx.rank < partner);
                merge_acc += MPI_Wtime() - t0;
                swapped_this_iteration = 1;
            }
        }

        // --- Odd Phase ---
        partner = (ctx.rank % 2 != 0) ? (ctx.rank + 1) : (ctx.rank - 1);
        if (partner >= 0 && partner < ctx.size)
        {
            int partner_n = (ctx.rank % 2 != 0) ? next_n : prev_n;

            Boundary my_b = {}, partner_b = {};
            if (ctx.local_n > 0)
                my_b = {data.front(), data.back()};
            t0 = MPI_Wtime();
            MPI_Sendrecv(&my_b, 2, MPI_FLOAT, partner, 2, &partner_b, 2, MPI_FLOAT, partner, 2, ctx.comm, MPI_STATUS_IGNORE);
            comm_probe_acc += MPI_Wtime() - t0;

            bool needs_merge = false;
            if (ctx.local_n > 0 && partner_n > 0)
            {
                if ((ctx.rank < partner && my_b.last > partner_b.first) || (ctx.rank > partner && my_b.first < partner_b.last))
                {
                    needs_merge = true;
                }
            }
            if (needs_merge)
            {
                t0 = MPI_Wtime();
                MPI_Sendrecv(data.data(), ctx.local_n, MPI_FLOAT, partner, 3, recv_buf.data(), partner_n, MPI_FLOAT, partner, 3, ctx.comm, MPI_STATUS_IGNORE);
                comm_data_acc += MPI_Wtime() - t0;

                t0 = MPI_Wtime();
                merge_and_swap(data, recv_buf, partner_n, ctx.rank < partner);
                merge_acc += MPI_Wtime() - t0;
                swapped_this_iteration = 1;
            }
        }

        int total_swapped = 0;
        MPI_Allreduce(&swapped_this_iteration, &total_swapped, 1, MPI_INT, MPI_LOR, ctx.comm);
        if (total_swapped == 0)
        {
            break;
        }
    }
    timer.comm_probe = comm_probe_acc;
    timer.comm_data = comm_data_acc;
    timer.merge = merge_acc;

    t0 = MPI_Wtime();
    MPI_File out_file;
    MPI_File_open(ctx.comm, output_path.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out_file);
    if (ctx.local_n > 0)
    {
        MPI_File_write_at(out_file, ctx.offset * sizeof(float), data.data(), ctx.local_n, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&out_file);
    timer.io_write = MPI_Wtime() - t0;
    timer.total = MPI_Wtime() - t_start;

    if (do_profile)
    {
        double max_total = 0.0, max_read = 0.0, max_write = 0.0, max_sort = 0.0, max_merge = 0.0;
        double max_comm_p = 0.0, max_comm_d = 0.0;

        MPI_Reduce(&timer.total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, ctx.comm);
        MPI_Reduce(&timer.io_read, &max_read, 1, MPI_DOUBLE, MPI_MAX, 0, ctx.comm);
        MPI_Reduce(&timer.io_write, &max_write, 1, MPI_DOUBLE, MPI_MAX, 0, ctx.comm);
        MPI_Reduce(&timer.local_sort, &max_sort, 1, MPI_DOUBLE, MPI_MAX, 0, ctx.comm);
        MPI_Reduce(&timer.merge, &max_merge, 1, MPI_DOUBLE, MPI_MAX, 0, ctx.comm);
        MPI_Reduce(&timer.comm_probe, &max_comm_p, 1, MPI_DOUBLE, MPI_MAX, 0, ctx.comm);
        MPI_Reduce(&timer.comm_data, &max_comm_d, 1, MPI_DOUBLE, MPI_MAX, 0, ctx.comm);

        if (ctx.rank == 0)
        {
            // --- START: 已修改的輸出區塊 ---
            double total_io = max_read + max_write;
            double total_comm = max_comm_p + max_comm_d;
            double total_cpu = max_sort + max_merge;

            printf("\n--- Performance Profile ---\n");
            printf("Total Time:        %.6f s\n", max_total);
            if (max_total > 1e-9)
            {

                printf("\n[Summary for Report Plots]\n");
                printf("  - IO Time:       %.6f s (%.1f%%)\n", total_io, total_io / max_total * 100);
                printf("  - Comm Time:     %.6f s (%.1f%%)\n", total_comm, total_comm / max_total * 100);
                printf("  - CPU Time:      %.6f s (%.1f%%)\n", total_cpu, total_cpu / max_total * 100);

                printf("\n[Detailed Breakdown]\n");
                printf("  - I/O Read:      %.6f s\n", max_read);
                printf("  - I/O Write:     %.6f s\n", max_write);
                printf("  - Local Sort:    %.6f s\n", max_sort);
                printf("  - Merge:         %.6f s\n", max_merge);
                printf("  - Probe Comm:    %.6f s\n", max_comm_p);
                printf("  - Data Comm:     %.6f s\n", max_comm_d);
            }
            printf("---------------------------\n");
        }
    }

    MPI_Finalize();
    return 0;
}