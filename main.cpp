#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

#include "icp_3d.h"                   // 3D的ICP配准方法
#include "ndt_3d.h"                   // 3D的NDT配准方法


int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: ./main <source_pcd_file> <target_pcd_file>" << std::endl;
        return -1;
    }

    sad::CloudPtr source(new sad::PointCloudType), target(new sad::PointCloudType);
    pcl::io::loadPCDFile(argv[1], *source);
    pcl::io::loadPCDFile(argv[2], *target);

    sad::evaluate_and_call(
        [&]()
        {
            sad::Icp3d icp;
            icp.SetSource(source);
            icp.SetTarget(target);
            Sophus::SE3d pose;
            icp.AlignP2P(pose);
            std::cout << "icp p2p align success, pose: " << std::fixed << std::setprecision(8) << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << std::fixed << std::setprecision(8) << pose.translation().transpose() << std::endl;
            sad::CloudPtr source_trans(new sad::PointCloudType);
            pcl::transformPointCloud(*source, *source_trans, pose.matrix());
        },
        "ICP P2P", 1);

    /// 点到面
    sad::evaluate_and_call(
        [&]()
        {
            sad::Icp3d icp;
            icp.SetSource(source);
            icp.SetTarget(target);
            Sophus::SE3d pose;
            icp.AlignP2Plane(pose);
            std::cout << "icp p2plane align success, pose: " << std::fixed << std::setprecision(8) << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << std::fixed << std::setprecision(8) << pose.translation().transpose() << std::endl;
            sad::CloudPtr source_trans(new sad::PointCloudType);
            pcl::transformPointCloud(*source, *source_trans, pose.matrix());
        },
        "ICP P2Plane", 1);

    /// 点到线
    sad::evaluate_and_call(
        [&]()
        {
            sad::Icp3d icp;
            icp.SetSource(source);
            icp.SetTarget(target);
            Sophus::SE3d pose;
            icp.AlignP2Line(pose);
            std::cout << "icp p2line align success, pose: " << std::fixed << std::setprecision(8) << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << std::fixed << std::setprecision(8) << pose.translation().transpose() << std::endl;
            sad::CloudPtr source_trans(new sad::PointCloudType);
            pcl::transformPointCloud(*source, *source_trans, pose.matrix());
        },
        "ICP P2Line", 1);

    /// 第７章的NDT
    sad::evaluate_and_call(
        [&]()
        {
            sad::Ndt3d::Options options;
            options.voxel_size_ = 0.5;
            options.remove_centroid_ = true;
            options.nearby_type_ = sad::Ndt3d::NearbyType::CENTER;
            sad::Ndt3d ndt(options);
            ndt.SetSource(source);
            ndt.SetTarget(target);
            Sophus::SE3d pose;
            ndt.AlignNdt(pose);
            std::cout << "ndt align success, pose: " << std::fixed << std::setprecision(8) << pose.so3().unit_quaternion().coeffs().transpose()
                          << ", " << std::fixed << std::setprecision(8) << pose.translation().transpose() << std::endl;
                sad::CloudPtr source_trans(new sad::PointCloudType);
                pcl::transformPointCloud(*source, *source_trans, pose.matrix());
        },
        "NDT", 1);

    /// PCL ICP 作为备选
    sad::evaluate_and_call(
        [&]()
        {
            pcl::IterativeClosestPoint<sad::PointType, sad::PointType> icp_pcl;
            icp_pcl.setInputSource(source);
            icp_pcl.setInputTarget(target);
            sad::CloudPtr output_pcl(new sad::PointCloudType);
            icp_pcl.align(*output_pcl);
            Sophus::SE3f T = Sophus::SE3f(icp_pcl.getFinalTransformation());
            std::cout << "pose from icp pcl: " << std::fixed << std::setprecision(8) << T.so3().unit_quaternion().coeffs().transpose() << ", "
                      << std::fixed << std::setprecision(8) << T.translation().transpose() << std::endl;
        },
        "ICP PCL", 1);

    /// PCL NDT 作为备选
    sad::evaluate_and_call(
        [&]()
        {
            pcl::NormalDistributionsTransform<sad::PointType, sad::PointType> ndt_pcl;
            ndt_pcl.setInputSource(source);
            ndt_pcl.setInputTarget(target);
            ndt_pcl.setResolution(0.5);
            sad::CloudPtr output_pcl(new sad::PointCloudType);
            ndt_pcl.align(*output_pcl);
            Sophus::SE3f T = Sophus::SE3f(ndt_pcl.getFinalTransformation());
            std::cout << "pose from ndt pcl: " << std::fixed << std::setprecision(8) << T.so3().unit_quaternion().coeffs().transpose() << ", "
                      << T.translation().transpose() << ', trans: ' << std::fixed << std::setprecision(8) << ndt_pcl.getTransformationProbability() << std::endl;
        },
        "NDT PCL", 1);

    return 0;
}
