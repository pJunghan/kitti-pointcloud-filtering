// (헤더 포함은 동일, 생략하지 않고 유지)

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <boost/filesystem.hpp>
#include <omp.h>
#include <pcl/filters/passthrough.h>
#include <thread>
#include <chrono>
#include <atomic>

// 통계 변수
std::atomic<int> total_original_points(0);
std::atomic<int> total_filtered_points(0);

void readKITTIBin(const std::string& path, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file " << path << std::endl;
        return;
    }
    cloud->clear();

    float data[4];
    while (file.read(reinterpret_cast<char*>(data), sizeof(float) * 4))
    {
        pcl::PointXYZI point;
        point.x = data[0];
        point.y = data[1];
        point.z = data[2];
        point.intensity = data[3];
        cloud->points.push_back(point);
    }

    file.close();
    std::cout << "Loaded " << cloud->points.size() << " points from " << path << std::endl;
}

void downsampleAndSave(const std::string& input_file, const std::string& output_folder)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    readKITTIBin(input_file, cloud);

    // PassThrough 필터 - Z
    pcl::PointCloud<pcl::PointXYZI>::Ptr pass_filtered_z(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-3.0, 3.0);
    pass.filter(*pass_filtered_z);

    // X
    pcl::PointCloud<pcl::PointXYZI>::Ptr pass_filtered_x(new pcl::PointCloud<pcl::PointXYZI>);
    pass.setInputCloud(pass_filtered_z);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-10.0, 7.0);
    pass.filter(*pass_filtered_x);

    // Y
    pcl::PointCloud<pcl::PointXYZI>::Ptr pass_filtered_y(new pcl::PointCloud<pcl::PointXYZI>);
    pass.setInputCloud(pass_filtered_x);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-20.0, 20.0);
    pass.filter(*pass_filtered_y);

    // 노이즈 제거
    pcl::PointCloud<pcl::PointXYZI>::Ptr sor_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
    sor.setInputCloud(pass_filtered_y);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*sor_filtered);

    // 다운샘플링
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::VoxelGrid<pcl::PointXYZI> voxel;
    voxel.setInputCloud(sor_filtered);
    voxel.setLeafSize(0.05f, 0.05f, 0.05f);
    voxel.filter(*filtered);

    // 결과 저장
    boost::filesystem::path output_dir(output_folder);
    if (!boost::filesystem::exists(output_dir))
    {
        boost::filesystem::create_directories(output_dir);
        std::cout << "Created output folder: " << output_dir.string() << std::endl;
    }

    boost::filesystem::path input_path(input_file);
    boost::filesystem::path output_path = output_dir / input_path.filename().replace_extension(".pcd");

    pcl::io::savePCDFileBinary(output_path.string(), *filtered);
    std::cout << "Saved downsampled point cloud to " << output_path.string() << std::endl;

    total_original_points += cloud->size();
    total_filtered_points += filtered->size();
}

void processBatch(const std::vector<std::string>& files, const std::string& output_folder, int batch_size)
{
    omp_set_num_threads(8);

    for (size_t i = 0; i < files.size(); i += batch_size)
    {
        size_t end = std::min(i + batch_size, files.size());
        std::cout << "Processing files " << i << " to " << end - 1 << std::endl;

#pragma omp parallel for schedule(dynamic)
        for (int j = static_cast<int>(i); j < static_cast<int>(end); ++j)
        {
            downsampleAndSave(files[j], output_folder);
        }
    }
}

std::vector<std::string> getBinFiles(const std::string& folder_path)
{
    std::vector<std::string> files;
    boost::filesystem::path p(folder_path);

    if (!boost::filesystem::exists(p) || !boost::filesystem::is_directory(p))
    {
        std::cerr << "Invalid folder: " << folder_path << std::endl;
        return files;
    }

    for (const auto& entry : boost::filesystem::directory_iterator(p))
    {
        if (boost::filesystem::is_regular_file(entry.status()) && entry.path().extension() == ".bin")
        {
            files.push_back(entry.path().string());
            if (files.size() >= 200)
                break;
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

void processSequence(const std::string& folder_path, const std::string& output_folder, int batch_size)
{
    auto files = getBinFiles(folder_path);
    if (files.empty())
    {
        std::cerr << "No .bin files found in " << folder_path << std::endl;
        return;
    }
    processBatch(files, output_folder, batch_size);
    int removed_points = total_original_points - total_filtered_points;
    float removal_ratio = (float)removed_points / total_original_points * 100.0f;

    std::cout << "============================================" << std::endl;
    std::cout << "Total points before filtering: " << total_original_points << std::endl;
    std::cout << "Total points after filtering:  " << total_filtered_points << std::endl;
    std::cout << "Removed " << removed_points << " points (" << removal_ratio << "%)" << std::endl;
    std::cout << "============================================" << std::endl;

    total_original_points = 0;
    total_filtered_points = 0;
}

void visualizeComparisonStages(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& original,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& sor_filtered,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& voxel_filtered)
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Filtering Stages Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // 원본: 흰색
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> original_color(original, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZI>(original, original_color, "original");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original");

    // 노이즈 제거 후: 초록색
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> sor_color(sor_filtered, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZI>(sor_filtered, sor_color, "sor_filtered");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sor_filtered");

    // 복셀 다운샘플링 후: 빨간색
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> voxel_color(voxel_filtered, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZI>(voxel_filtered, voxel_color, "voxel_filtered");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "voxel_filtered");

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


int main()
{
    std::string base_path = "C:/dataset/sequences";
    int last_sequence = 1;
    int batch_size = 5;

    for (int seq = 0; seq < last_sequence; ++seq)
    {
        char seq_str[3];
        std::snprintf(seq_str, sizeof(seq_str), "%02d", seq);

        std::string input_folder = base_path + "/" + seq_str + "/velodyne";
        std::string output_folder = base_path + "/" + seq_str + "/velodyne_downsampled";

        std::cout << "Processing sequence: " << seq_str << std::endl;
        processSequence(input_folder, output_folder, batch_size);

        auto files = getBinFiles(input_folder);
        if (!files.empty())
        {
            pcl::PointCloud<pcl::PointXYZI>::Ptr last_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            readKITTIBin(files.back(), last_cloud);

            // 동일한 필터링
            pcl::PointCloud<pcl::PointXYZI>::Ptr pass_filtered(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::PassThrough<pcl::PointXYZI> pass;
            pass.setInputCloud(last_cloud);
            pass.setFilterFieldName("z");
            pass.setFilterLimits(-3.0, 3.0);
            pass.filter(*pass_filtered);

            pass.setInputCloud(pass_filtered);
            pass.setFilterFieldName("x");
            pass.setFilterLimits(-10.0, 7.0);
            pass.filter(*pass_filtered);

            pass.setInputCloud(pass_filtered);
            pass.setFilterFieldName("y");
            pass.setFilterLimits(-20.0, 20.0);
            pass.filter(*pass_filtered);

            pcl::PointCloud<pcl::PointXYZI>::Ptr sor_filtered(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
            sor.setInputCloud(pass_filtered);
            sor.setMeanK(50);
            sor.setStddevMulThresh(1.0);
            sor.filter(*sor_filtered);

            pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::VoxelGrid<pcl::PointXYZI> voxel;
            voxel.setInputCloud(sor_filtered);
            voxel.setLeafSize(0.05f, 0.05f, 0.05f);
            voxel.filter(*filtered);

            std::cout << "Visualizing last frame (before and after filtering)..." << std::endl;
            visualizeComparisonStages(last_cloud, sor_filtered, filtered);

        }
        else
        {
            std::cerr << "No files to visualize." << std::endl;
        }
    }

    return 0;
}
