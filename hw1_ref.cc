#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <mpi.h>

using namespace std;

// 全域變數避免重複分配記憶體
float *merge_buffer = nullptr;
int global_n, local_size;

// 高效的資料分配策略 (結合兩種方法的優點)
struct DataDistribution
{
    int elements_per_process;
    int remainder;
    int local_count;
    int offset;

    DataDistribution(int n, int rank, int size)
    {
        // 使用更精確的分配方式
        elements_per_process = n / size;
        remainder = n % size;

        // 前 remainder 個 processes 多分配一個元素
        if (rank < remainder)
        {
            local_count = elements_per_process + 1;
            offset = rank * local_count;
        }
        else
        {
            local_count = elements_per_process;
            offset = remainder * (elements_per_process + 1) +
                     (rank - remainder) * elements_per_process;
        }
    }
};

// 混合排序策略：小陣列用 insertion sort，大陣列用 std::sort
void hybridSort(vector<float> &arr)
{
    const int INSERTION_THRESHOLD = 16;

    if (arr.size() <= INSERTION_THRESHOLD)
    {
        // Insertion sort for small arrays
        for (int i = 1; i < arr.size(); i++)
        {
            float key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key)
            {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    }
    else
    {
        // std::sort for larger arrays
        sort(arr.begin(), arr.end());
    }
}

// 智慧型邊界檢查：避免不必要的資料交換
bool needExchange(const vector<float> &local_data, int neighbor_rank,
                  bool is_left_neighbor, MPI_Comm comm)
{
    if (local_data.empty())
        return false;

    float boundary_value;
    float neighbor_boundary;

    if (is_left_neighbor)
    {
        boundary_value = local_data[0]; // 我的最小值
    }
    else
    {
        boundary_value = local_data.back(); // 我的最大值
    }

    // 交換邊界值進行預檢查
    MPI_Sendrecv(&boundary_value, 1, MPI_FLOAT, neighbor_rank, 0,
                 &neighbor_boundary, 1, MPI_FLOAT, neighbor_rank, 0,
                 comm, MPI_STATUS_IGNORE);

    if (is_left_neighbor)
    {
        return boundary_value < neighbor_boundary; // 需要交換
    }
    else
    {
        return boundary_value > neighbor_boundary; // 需要交換
    }
}

// 最佳化的合併函數
bool optimizedMerge(vector<float> &local_data, vector<float> &neighbor_data,
                    bool keep_smaller)
{
    int local_size = local_data.size();
    int neighbor_size = neighbor_data.size();

    if (local_size == 0 || neighbor_size == 0)
        return false;

    // 早期返回檢查
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

    // 使用全域 buffer 避免重複分配
    if (!merge_buffer)
    {
        merge_buffer = new float[local_size];
    }

    int i = 0, j = 0, k = 0;
    bool has_change = false;

    if (keep_smaller)
    {
        // 合併取較小的部分
        while (k < local_size)
        {
            if (i >= local_size)
            {
                merge_buffer[k++] = neighbor_data[j++];
            }
            else if (j >= neighbor_size)
            {
                merge_buffer[k++] = local_data[i++];
            }
            else if (local_data[i] <= neighbor_data[j])
            {
                merge_buffer[k++] = local_data[i++];
            }
            else
            {
                merge_buffer[k++] = neighbor_data[j++];
            }
        }
    }
    else
    {
        // 反向合併取較大的部分
        i = local_size - 1;
        j = neighbor_size - 1;
        k = local_size - 1;

        while (k >= 0)
        {
            if (i < 0)
            {
                merge_buffer[k--] = neighbor_data[j--];
            }
            else if (j < 0)
            {
                merge_buffer[k--] = local_data[i--];
            }
            else if (local_data[i] >= neighbor_data[j])
            {
                merge_buffer[k--] = local_data[i--];
            }
            else
            {
                merge_buffer[k--] = neighbor_data[j--];
            }
        }
    }

    // 檢查是否有變化並複製回去
    for (int idx = 0; idx < local_size; idx++)
    {
        if (local_data[idx] != merge_buffer[idx])
        {
            has_change = true;
        }
        local_data[idx] = merge_buffer[idx];
    }

    return has_change;
}

// 平行 Odd-Even Sort 主函數
void parallelOddEvenSort(vector<float> &local_data, int rank, int size, MPI_Comm comm)
{
    // 先對本地資料排序
    hybridSort(local_data);

    vector<float> neighbor_data;
    bool global_sorted = false;
    int max_iterations = size; // 理論上界

    // 計算鄰居的資料大小
    DataDistribution dist(global_n, rank + 1, size);
    int right_neighbor_size = (rank + 1 < size) ? dist.local_count : 0;

    dist = DataDistribution(global_n, rank - 1, size);
    int left_neighbor_size = (rank - 1 >= 0) ? dist.local_count : 0;

    for (int iteration = 0; iteration < max_iterations && !global_sorted; iteration++)
    {
        bool local_changed = false;

        // Even phase
        if (rank % 2 == 0)
        {
            // Even ranks 與右鄰居交換
            if (rank + 1 < size && !local_data.empty())
            {
                if (needExchange(local_data, rank + 1, false, comm))
                {
                    neighbor_data.resize(right_neighbor_size);
                    MPI_Sendrecv(local_data.data(), local_data.size(), MPI_FLOAT, rank + 1, 0,
                                 neighbor_data.data(), right_neighbor_size, MPI_FLOAT, rank + 1, 0,
                                 comm, MPI_STATUS_IGNORE);

                    if (optimizedMerge(local_data, neighbor_data, true))
                    {
                        local_changed = true;
                    }
                }
            }
        }
        else
        {
            // Odd ranks 與左鄰居交換
            if (rank - 1 >= 0 && !local_data.empty())
            {
                if (needExchange(local_data, rank - 1, true, comm))
                {
                    neighbor_data.resize(left_neighbor_size);
                    MPI_Sendrecv(local_data.data(), local_data.size(), MPI_FLOAT, rank - 1, 0,
                                 neighbor_data.data(), left_neighbor_size, MPI_FLOAT, rank - 1, 0,
                                 comm, MPI_STATUS_IGNORE);

                    if (optimizedMerge(local_data, neighbor_data, false))
                    {
                        local_changed = true;
                    }
                }
            }
        }

        // Odd phase
        if (rank % 2 == 1)
        {
            // Odd ranks 與右鄰居交換
            if (rank + 1 < size && !local_data.empty())
            {
                if (needExchange(local_data, rank + 1, false, comm))
                {
                    neighbor_data.resize(right_neighbor_size);
                    MPI_Sendrecv(local_data.data(), local_data.size(), MPI_FLOAT, rank + 1, 0,
                                 neighbor_data.data(), right_neighbor_size, MPI_FLOAT, rank + 1, 0,
                                 comm, MPI_STATUS_IGNORE);

                    if (optimizedMerge(local_data, neighbor_data, true))
                    {
                        local_changed = true;
                    }
                }
            }
        }
        else
        {
            // Even ranks 與左鄰居交換
            if (rank - 1 >= 0 && rank != 0 && !local_data.empty())
            {
                if (needExchange(local_data, rank - 1, true, comm))
                {
                    neighbor_data.resize(left_neighbor_size);
                    MPI_Sendrecv(local_data.data(), local_data.size(), MPI_FLOAT, rank - 1, 0,
                                 neighbor_data.data(), left_neighbor_size, MPI_FLOAT, rank - 1, 0,
                                 comm, MPI_STATUS_IGNORE);

                    if (optimizedMerge(local_data, neighbor_data, false))
                    {
                        local_changed = true;
                    }
                }
            }
        }

        // 全域收斂檢測
        MPI_Allreduce(&local_changed, &global_sorted, 1, MPI_C_BOOL, MPI_LOR, comm);
        global_sorted = !global_sorted; // 如果沒有人改變，就是排序完成
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

    // 智慧型資料分配
    DataDistribution dist(global_n, rank, size);
    vector<float> local_data(dist.local_count);

    // MPI-IO 讀取
    MPI_File infile;
    MPI_File_open(MPI_COMM_WORLD, input_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &infile);
    if (dist.local_count > 0)
    {
        MPI_File_read_at(infile, dist.offset * sizeof(float),
                         local_data.data(), dist.local_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&infile);

    // 執行平行 Odd-Even Sort
    parallelOddEvenSort(local_data, rank, size, MPI_COMM_WORLD);

    // MPI-IO 寫入
    MPI_File outfile;
    MPI_File_open(MPI_COMM_WORLD, output_file.c_str(),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile);
    if (dist.local_count > 0)
    {
        MPI_File_write_at(outfile, dist.offset * sizeof(float),
                          local_data.data(), dist.local_count, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&outfile);

    // 清理全域資源
    if (merge_buffer)
    {
        delete[] merge_buffer;
    }

    MPI_Finalize();
    return 0;
}