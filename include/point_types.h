#pragma once

#include <Eigen/Dense>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


namespace sad {
    // 矢量哈希
    template <int N>
    struct hash_vec {
        inline size_t operator()(const Eigen::Matrix<int, N, 1>& v) const;
    };

    /// @see Optimized Spatial Hashing for Collision Detection of Deformable Objects, Matthias Teschner et. al., VMV 2003
    template <>
    inline size_t hash_vec<2>::operator()(const Eigen::Matrix<int, 2, 1>& v) const {
        return size_t(((v[0] * 73856093) ^ (v[1] * 471943)) % 10000000);
    }

    template <>
    inline size_t hash_vec<3>::operator()(const Eigen::Matrix<int, 3, 1>& v) const {
        return size_t(((v[0] * 73856093) ^ (v[1] * 471943) ^ (v[2] * 83492791)) % 10000000);
    }

    // 定义系统中用到的点和点云类型
    using PointType = pcl::PointXYZ;
    using PointCloudType = pcl::PointCloud<PointType>;
    using CloudPtr = PointCloudType::Ptr;

    // 点云到Eigen的常用的转换函数
    inline Eigen::Vector3f ToVec3f(const PointType& pt) { return pt.getVector3fMap(); }
    inline Eigen::Vector3d ToVec3d(const PointType& pt) { return pt.getVector3fMap().cast<double>(); }

    template <typename S>
    inline PointType ToPointType(const Eigen::Matrix<S, 3, 1>& pt) {
        PointType p;
        p.x = pt.x();
        p.y = pt.y();
        p.z = pt.z();
        return p;
    }

}  // namespace sad

