#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <cstdlib>
#include <string>
#include <boost/sort/spreadsort/spreadsort.hpp>

// ---- Profiling helpers ----
struct Timers
{
    double t_total{0.0};
    double t_io_read{0.0};
    double t_io_write{0.0};
    double t_sort{0.0};
    double t_comm{0.0};
    double t_merge{0.0};
};
static inline double now() { return MPI_Wtime(); }

using namespace std;

int global_n;
constexpr int DATA_TAG = 8;

static inline int compute_partner(int rank, int phase)
{
    // 展開邏輯，不用位運算
    bool is_even_phase = (phase % 2 == 0);
    bool is_even_rank = (rank % 2 == 0);

    return (is_even_phase == is_even_rank) ? rank + 1 : rank - 1;
}

static inline bool merge_keep_half(std::vector<float> &me,
                                   const std::vector<float> &other,
                                   bool keep_smaller,
                                   std::vector<float> &merge_buffer)
{
    const size_t keep = me.size();
    const size_t total = keep + other.size();

    // 預先調整長度（不改變 capacity）
    if (merge_buffer.size() != total)
    {
        merge_buffer.resize(total);
    }

    // 直接把 merge 輸出到緩衝區起點（零 push_back/迭代器包裝）
    std::merge(me.begin(), me.end(), other.begin(), other.end(),
               merge_buffer.begin());

    bool changed = false;
    if (keep_smaller)
    {
        // 拷貝前半
        for (size_t i = 0; i < keep; ++i)
        {
            float v = merge_buffer[i];
            if (!changed && me[i] != v)
                changed = true;
            me[i] = v;
        }
    }
    else
    {
        // 拷貝後半
        size_t start = total - keep;
        for (size_t i = 0; i < keep; ++i)
        {
            float v = merge_buffer[start + i];
            if (!changed && me[i] != v)
                changed = true;
            me[i] = v;
        }
    }
    return changed;
}

void hybridSort(vector<float> &arr)
{
    const int INSERTION_THRESHOLD = 32;
    if ((int)arr.size() <= INSERTION_THRESHOLD)
    {
        for (int i = 1; i < (int)arr.size(); i++)
        {
            float key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key)
            {
                arr[j + 1] = arr[j];
                --j;
            }
            arr[j + 1] = key;
        }
    }
    else
    {
        boost::sort::spreadsort::spreadsort(arr.begin(), arr.end());
    }
}

struct DataDistribution
{
    int local_count, offset;
    DataDistribution(int n, int rank, int size)
    {
        int base_size = n / size;
        int extra = n - (base_size * size);

        int has_extra = (rank < extra) ? 1 : 0;
        local_count = base_size + has_extra;

        offset = rank * base_size + ((rank < extra) ? rank : extra);
    }
};

void parallelOddEvenSort(std::vector<float> &local_data,
                         int rank, int size,
                         const std::vector<int> &counts,
                         MPI_Comm comm,
                         Timers &T)
{
    // 本地排序
    double t0 = now();
    hybridSort(local_data);
    T.t_sort += (now() - t0);

    // 預估左右鄰大小，配置可重用緩衝
    const int left_rank = rank - 1;
    const int right_rank = rank + 1;
    const int left_count = (left_rank >= 0) ? counts[left_rank] : 0;
    const int right_count = (right_rank < size) ? counts[right_rank] : 0;

    std::vector<float> neighbor(std::max(left_count, right_count));
    //
    std::vector<float> merge_buffer;
    merge_buffer.reserve(std::max(left_count, right_count) + counts[rank]);

    int pair_changed_local = 0; // 累積一對 even+odd 的變動

    for (int phase = 0; phase < size; ++phase)
    {
        const int partner = compute_partner(rank, phase);
        int local_changed = 0;

        if (partner >= 0 && partner < size)
        {
            const int partner_count = counts[partner];

            // zero-length
            if (!local_data.empty() && partner_count > 0)
            {
                if ((int)neighbor.size() != partner_count)
                {
                    neighbor.resize(partner_count);
                }

                double tc0 = now();
                MPI_Sendrecv(local_data.data(), (int)local_data.size(), MPI_FLOAT, partner, DATA_TAG,
                             neighbor.data(), partner_count, MPI_FLOAT, partner, DATA_TAG,
                             comm, MPI_STATUS_IGNORE);
                T.t_comm += (now() - tc0);

                const bool keep_smaller = (rank < partner);

                // 提前檢查是否需要合併
                const float my_min = local_data.front();
                const float my_max = local_data.back();
                const float nb_min = neighbor.front();
                const float nb_max = neighbor.back();

                if ((keep_smaller && my_max <= nb_min) ||
                    (!keep_smaller && my_min >= nb_max))
                {
                    local_changed = 0;
                }
                else
                {
                    double tm0 = now();
                    local_changed = merge_keep_half(local_data, neighbor, keep_smaller, merge_buffer) ? 1 : 0;
                    T.t_merge += (now() - tm0);
                }
            }
        }

        pair_changed_local |= local_changed;

        if (phase & 1)
        {
            int pair_changed_any = 0;
            MPI_Allreduce(&pair_changed_local, &pair_changed_any, 1, MPI_INT, MPI_LOR, comm);
            if (pair_changed_any == 0)
                break;              // 這對 even+odd 都沒變，提前停
            pair_changed_local = 0; // 重置，下一對 even+odd
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4)
    {
        if (rank == 0)
            cerr << "Usage: " << argv[0] << " N input_file output_file [--profile] [--verify]\n";
        MPI_Finalize();
        return 1;
    }

    // 參數與旗標
    bool do_profile = false;
    bool do_verify = false;
    // 環境變數也可啟用 profile
    if (const char *env = std::getenv("HW1_PROFILE"))
    {
        if (std::string(env) == "1")
            do_profile = true;
    }
    // 額外旗標
    for (int i = 4; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--profile")
            do_profile = true;
        else if (a == "--verify")
            do_verify = true;
    }

    double T0 = now();

    global_n = atoi(argv[1]);
    string input_file = argv[2];
    string output_file = argv[3];

    // === 建立 counts[] 與 offsets[] ===
    vector<int> counts(size), offsets(size);
    int base = global_n / size;
    int rem = global_n % size;
    int sum = 0;
    for (int i = 0; i < size; i++)
    {
        counts[i] = base + (i < rem ? 1 : 0);
        offsets[i] = sum;
        sum += counts[i];
    }

    // local 資料
    vector<float> local_data(counts[rank]);

    // 讀檔
    double tr0 = now();
    MPI_File infile;
    MPI_File_open(MPI_COMM_WORLD, input_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);
    if (counts[rank] > 0)
    {
        MPI_File_read_at(infile, (MPI_Offset)offsets[rank] * sizeof(float),
                         local_data.data(), counts[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&infile);
    double t_io_read_local = now() - tr0;

    // 排序（帶 Timers）
    Timers T{};
    parallelOddEvenSort(local_data, rank, size, counts, MPI_COMM_WORLD, T);

    // 寫檔
    double tw0 = now();
    MPI_File outfile;
    MPI_File_open(MPI_COMM_WORLD, output_file.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile);
    if (counts[rank] > 0)
    {
        MPI_File_write_at(outfile, (MPI_Offset)offsets[rank] * sizeof(float),
                          local_data.data(), counts[rank], MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&outfile);
    double t_io_write_local = now() - tw0;

    T.t_io_read = t_io_read_local;
    T.t_io_write = t_io_write_local;
    T.t_total = now() - T0;

    // Profile 輸出（取各 rank 的「最大值」代表瓶頸）
    if (do_profile)
    {
        auto reduce_max = [&](double x)
        {
            double g = 0.0;
            MPI_Reduce(&x, &g, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            return g;
        };
        double g_total = reduce_max(T.t_total);
        double g_read = reduce_max(T.t_io_read);
        double g_write = reduce_max(T.t_io_write);
        double g_sort = reduce_max(T.t_sort);
        double g_comm = reduce_max(T.t_comm);
        double g_merge = reduce_max(T.t_merge);
        double tick = MPI_Wtick();

        if (rank == 0)
        {
            std::cout.setf(std::ios::fixed);
            std::cout.precision(6);
            std::cout << "[PROFILE] MPI_Wtick = " << tick << " s\n";
            std::cout << "[PROFILE] total      = " << g_total << " s\n";
            std::cout << "[PROFILE] io_read    = " << g_read << " s\n";
            std::cout << "[PROFILE] io_write   = " << g_write << " s\n";
            std::cout << "[PROFILE] local_sort = " << g_sort << " s\n";
            std::cout << "[PROFILE] comm       = " << g_comm << " s\n";
            std::cout << "[PROFILE] merge      = " << g_merge << " s\n";
        }
    }

    MPI_Finalize();
    return 0;
}
