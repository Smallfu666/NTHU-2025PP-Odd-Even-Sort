#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <mpi.h>

using namespace std;

// 全域變數 - 優化記憶體使用
float *merge_buffer = nullptr;
float *radix_buffer = nullptr;
int global_n;

// IEEE 754 浮點數 Radix Sort 優化
inline uint32_t floatToRadixKey(float f)
{
    uint32_t key = *(uint32_t *)&f;
    // 處理符號位：正數 flip sign bit，負數 flip 全部
    return (key & 0x80000000) ? ~key : (key | 0x80000000);
}

inline float radixToFloat(uint32_t key)
{
    key ^= ((key >> 31) - 1) | 0x80000000;
    return *(float *)&key;
}

// 4-pass Radix Sort - O(n) 複雜度
void radixSort(vector<float> &arr)
{
    if (arr.size() <= 1)
        return;

    const int n = arr.size();
    if (!radix_buffer)
    {
        radix_buffer = new float[n];
    }

    uint32_t *keys = new uint32_t[n];
    uint32_t *temp_keys = new uint32_t[n];

    // 轉換為 radix keys
    for (int i = 0; i < n; i++)
    {
        keys[i] = floatToRadixKey(arr[i]);
    }

    // 4-pass counting sort
    for (int pass = 0; pass < 4; pass++)
    {
        const int shift = pass * 8;
        int count[256] = {0};

        // 計數
        for (int i = 0; i < n; i++)
        {
            count[(keys[i] >> shift) & 0xFF]++;
        }

        // 前綴和
        int sum = 0;
        for (int i = 0; i < 256; i++)
        {
            int temp = count[i] + sum;
            count[i] = sum - 1;
            sum = temp;
        }

        // 分散
        for (int i = 0; i < n; i++)
        {
            int bucket = (keys[i] >> shift) & 0xFF;
            temp_keys[++count[bucket]] = keys[i];
        }

        // 交換 buffer
        swap(keys, temp_keys);
    }

    // 轉回 float
    for (int i = 0; i < n; i++)
    {
        arr[i] = radixToFloat(keys[i]);
    }

    delete[] keys;
    delete[] temp_keys;
}

// 超級優化的資料分配策略
struct OptimizedDataDistribution
{
    int local_count;
    int offset;
    int left_neighbor_count;
    int right_neighbor_count;

    OptimizedDataDistribution(int n, int rank, int size)
    {
        // 使用 double 避免精度問題
        int base_count = n / size;
        int remainder = n % size;

        // 前 remainder 個 processes 多分配一個
        if (rank < remainder)
        {
            local_count = base_count + 1;
            offset = rank * local_count;
        }
        else
        {
            local_count = base_count;
            offset = remainder * (base_count + 1) + (rank - remainder) * base_count;
        }

        // 預計算鄰居大小
        int left_rank = rank - 1;
        int right_rank = rank + 1;

        left_neighbor_count = (left_rank >= 0) ? (left_rank < remainder ? base_count + 1 : base_count) : 0;

        right_neighbor_count = (right_rank < size) ? (right_rank < remainder ? base_count + 1 : base_count) : 0;
    }
};

// 超級優化的邊界檢查 - 減少通訊次數
bool ultraFastNeedExchange(const vector<float> &local_data,
                           int neighbor_rank, bool is_left_neighbor,
                           float &cached_boundary, MPI_Comm comm)
{
    if (local_data.empty())
        return false;

    // 使用快取避免重複計算
    float boundary_value = is_left_neighbor ? local_data[0] : local_data.back();
    if (boundary_value == cached_boundary)
        return false; // 假設上次檢查過

    float neighbor_boundary;
    MPI_Sendrecv(&boundary_value, 1, MPI_FLOAT, neighbor_rank, 0,
                 &neighbor_boundary, 1, MPI_FLOAT, neighbor_rank, 0,
                 comm, MPI_STATUS_IGNORE);

    cached_boundary = boundary_value;

    return is_left_neighbor ? (boundary_value < neighbor_boundary) : (boundary_value > neighbor_boundary);
}

// 神級合併函數 - 零複製優化
bool godlikeMerge(vector<float> &local_data, const vector<float> &neighbor_data,
                  bool keep_smaller, float *global_buffer)
{
    const int local_size = local_data.size();
    const int neighbor_size = neighbor_data.size();

    if (local_size == 0 || neighbor_size == 0)
        return false;

    // 超級早期返回檢查
    if (keep_smaller)
    {
        if (local_data.back() <= neighbor_data[0])
            return false;
    }
    else
    {
        if (local_data[0] >= neighbor_data.back())
            return false;
    }

    // 使用傳入的 buffer 避免重複分配
    float *temp = global_buffer;

    int i = 0, j = 0, k = 0;
    bool has_change = false;

    if (keep_smaller)
    {
        // 優化版合併 - 減少條件判斷
        const int *i_end = &local_size;
        const int *j_end = &neighbor_size;

        while (k < local_size)
        {
            const bool i_done = i >= local_size;
            const bool j_done = j >= neighbor_size;

            if (i_done)
            {
                temp[k++] = neighbor_data[j++];
            }
            else if (j_done)
            {
                temp[k++] = local_data[i++];
            }
            else
            {
                temp[k++] = (local_data[i] <= neighbor_data[j]) ? local_data[i++] : neighbor_data[j++];
            }
        }
    }
    else
    {
        // 反向合併優化
        i = local_size - 1;
        j = neighbor_size - 1;
        k = local_size - 1;

        while (k >= 0)
        {
            const bool i_done = i < 0;
            const bool j_done = j < 0;

            if (i_done)
            {
                temp[k--] = neighbor_data[j--];
            }
            else if (j_done)
            {
                temp[k--] = local_data[i--];
            }
            else
            {
                temp[k--] = (local_data[i] >= neighbor_data[j]) ? local_data[i--] : neighbor_data[j--];
            }
        }
    }

    // 單遍檢查變化並複製
    for (int idx = 0; idx < local_size; idx++)
    {
        has_change |= (local_data[idx] != temp[idx]);
        local_data[idx] = temp[idx];
    }

    return has_change;
}

// 超級平行 Odd-Even Sort - 極致優化
void ultraParallelOddEvenSort(vector<float> &local_data, int rank, int size, MPI_Comm comm)
{
    // 使用 Radix Sort 預排序 - O(n) vs O(n log n)
    radixSort(local_data);

    OptimizedDataDistribution dist(global_n, rank, size);
    vector<float> neighbor_data;
    float cached_left_boundary = -FLT_MAX;
    float cached_right_boundary = FLT_MAX;

    // 動態收斂檢測 - 結合多種策略
    int consecutive_no_change = 0;
    const int MAX_CONSECUTIVE_NO_CHANGE = 3;
    int iteration = 0;
    const int MAX_ITERATIONS = size * 2; // 更寬鬆的上界

    while (iteration < MAX_ITERATIONS && consecutive_no_change < MAX_CONSECUTIVE_NO_CHANGE)
    {
        bool local_changed = false;

        // Even Phase - 超級優化
        if ((rank & 1) == 0)
        { // 使用 bitwise 檢查 even
            if (rank + 1 < size && !local_data.empty())
            {
                if (ultraFastNeedExchange(local_data, rank + 1, false, cached_right_boundary, comm))
                {
                    neighbor_data.resize(dist.right_neighbor_count);
                    // 單次大訊息交換
                    MPI_Sendrecv(local_data.data(), local_data.size(), MPI_FLOAT, rank + 1, 0,
                                 neighbor_data.data(), dist.right_neighbor_count, MPI_FLOAT, rank + 1, 0,
                                 comm, MPI_STATUS_IGNORE);

                    if (godlikeMerge(local_data, neighbor_data, true, merge_buffer))
                    {
                        local_changed = true;
                    }
                }
            }
        }
        else
        {
            if (rank - 1 >= 0 && !local_data.empty())
            {
                if (ultraFastNeedExchange(local_data, rank - 1, true, cached_left_boundary, comm))
                {
                    neighbor_data.resize(dist.left_neighbor_count);
                    MPI_Sendrecv(local_data.data(), local_data.size(), MPI_FLOAT, rank - 1, 0,
                                 neighbor_data.data(), dist.left_neighbor_count, MPI_FLOAT, rank - 1, 0,
                                 comm, MPI_STATUS_IGNORE);

                    if (godlikeMerge(local_data, neighbor_data, false, merge_buffer))
                    {
                        local_changed = true;
                    }
                }
            }
        }

        // Odd Phase - 對稱優化
        if ((rank & 1) == 1)
        { // 使用 bitwise 檢查 odd
            if (rank + 1 < size && !local_data.empty())
            {
                if (ultraFastNeedExchange(local_data, rank + 1, false, cached_right_boundary, comm))
                {
                    neighbor_data.resize(dist.right_neighbor_count);
                    MPI_Sendrecv(local_data.data(), local_data.size(), MPI_FLOAT, rank + 1, 0,
                                 neighbor_data.data(), dist.right_neighbor_count, MPI_FLOAT, rank + 1, 0,
                                 comm, MPI_STATUS_IGNORE);

                    if (godlikeMerge(local_data, neighbor_data, true, merge_buffer))
                    {
                        local_changed = true;
                    }
                }
            }
        }
        else
        {
            if (rank - 1 >= 0 && rank != 0 && !local_data.empty())
            {
                if (ultraFastNeedExchange(local_data, rank - 1, true, cached_left_boundary, comm))
                {
                    neighbor_data.resize(dist.left_neighbor_count);
                    MPI_Sendrecv(local_data.data(), local_data.size(), MPI_FLOAT, rank - 1, 0,
                                 neighbor_data.data(), dist.left_neighbor_count, MPI_FLOAT, rank - 1, 0,
                                 comm, MPI_STATUS_IGNORE);

                    if (godlikeMerge(local_data, neighbor_data, false, merge_buffer))
                    {
                        local_changed = true;
                    }
                }
            }
        }

        // 智慧收斂檢測 - 結合局部和全域
        bool global_changed;
        MPI_Allreduce(&local_changed, &global_changed, 1, MPI_C_BOOL, MPI_LOR, comm);

        if (global_changed)
        {
            consecutive_no_change = 0;
        }
        else
        {
            consecutive_no_change++;
        }

        iteration++;
    }
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

    global_n = atoi(argv[1]);
    string input_file = argv[2];
    string output_file = argv[3];

    // 超級優化資料分配
    OptimizedDataDistribution dist(global_n, rank, size);
    vector<float> local_data(dist.local_count);

    // 預分配全域 buffer
    if (dist.local_count > 0)
    {
        merge_buffer = new float[dist.local_count];
    }

    // MPI-IO 讀取 - 優化版
    MPI_File infile;
    MPI_File_open(MPI_COMM_WORLD, input_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);
    if (dist.local_count > 0)
    {
        MPI_File_read_at(infile, dist.offset * sizeof(float),
                         local_data.data(), dist.local_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&infile);

    // 執行超級平行 Odd-Even Sort
    ultraParallelOddEvenSort(local_data, rank, size, MPI_COMM_WORLD);

    // MPI-IO 寫入 - 優化版
    MPI_File outfile;
    MPI_File_open(MPI_COMM_WORLD, output_file.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile);
    if (dist.local_count > 0)
    {
        MPI_File_write_at(outfile, dist.offset * sizeof(float),
                          local_data.data(), dist.local_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&outfile);

    // 超級清理
    if (merge_buffer)
        delete[] merge_buffer;
    if (radix_buffer)
        delete[] radix_buffer;

    MPI_Finalize();
    return 0;
}