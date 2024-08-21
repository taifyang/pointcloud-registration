#include "icp_3d.h"

#include <execution>

namespace sad {

bool Icp3d::AlignP2P(Sophus::SE3d& init_pose) {
    std::cout << "aligning with point to point" << std::endl;
    assert(target_ != nullptr && source_ != nullptr);

    Sophus::SE3d pose = init_pose;
    if (!options_.use_initial_translation_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
    }

    // 对点的索引，预先生成
    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    // 我们来写一些并发代码
    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(index.size());
    std::vector<Eigen::Vector3d> errors(index.size());

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        // gauss-newton 迭代 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Eigen::Vector3d qs = pose * q;  // 转换之后的q
            std::vector<int> nn;
            kdtree_->GetClosestPoint(ToPointType(qs), nn, 1);

            if (!nn.empty()) {
                Eigen::Vector3d p = ToVec3d(target_->points[nn[0]]);
                double dis2 = (p - qs).squaredNorm();
                if (dis2 > options_.max_nn_distance_) {
                    // 点离的太远了不要
                    effect_pts[idx] = false;
                    return;
                }

                effect_pts[idx] = true;

                // build residual
                Eigen::Vector3d e = p - qs;
                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = pose.so3().matrix() * Sophus::SO3d::hat(q);
                J.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

                jacobians[idx] = J;
                errors[idx] = e;
            } else {
                effect_pts[idx] = false;
            }
        });

        // 累加Hessian和error,计算dx
        // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        double total_res = 0;
        int effective_num = 0;
        auto H_and_err = std::accumulate(
            index.begin(), index.end(), std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 1>::Zero()),
            [&jacobians, &errors, &effect_pts, &total_res, &effective_num](const std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>& pre,
                                                                           int idx) -> std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> {
                if (!effect_pts[idx]) {
                    return pre;
                } else {
                    total_res += errors[idx].dot(errors[idx]);
                    effective_num++;
                    return std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(pre.first + jacobians[idx].transpose() * jacobians[idx],
                                                   pre.second - jacobians[idx].transpose() * errors[idx]);
                }
            });

        if (effective_num < options_.min_effective_pts_) {
            std::cout << "effective num too small: " << effective_num << std::endl;
            return false;
        }

        Eigen::Matrix<double, 6, 6> H = H_and_err.first;
        Eigen::Matrix<double, 6, 1> err = H_and_err.second;

        Eigen::Matrix<double, 6, 1> dx = H.inverse() * err;
        pose.so3() = pose.so3() * Sophus::SO3d::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        // 更新
        std::cout << "iter " << iter << " total res: " << total_res << ", eff: " << effective_num
                  << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm() << std::endl;

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            std::cout << "iter " << iter << " pose error: " << pose_error << std::endl;
        }

        if (dx.norm() < options_.eps_) {
            std::cout << "converged, dx = " << dx.transpose() << std::endl;
            break;
        }
    }

    init_pose = pose;
    return true;
}

bool Icp3d::AlignP2Plane(Sophus::SE3d& init_pose) {
    std::cout << "aligning with point to plane" << std::endl;
    assert(target_ != nullptr && source_ != nullptr);
    // 整体流程与p2p一致，读者请关注变化部分

    Sophus::SE3d pose = init_pose;
    if (!options_.use_initial_translation_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
    }

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 1, 6>> jacobians(index.size());
    std::vector<double> errors(index.size());

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        // gauss-newton 迭代 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Eigen::Vector3d qs = pose * q;  // 转换之后的q
            std::vector<int> nn;
            kdtree_->GetClosestPoint(ToPointType(qs), nn, 5);  // 这里取5个最近邻
            if (nn.size() > 3) {
                // convert to eigen
                std::vector<Eigen::Vector3d> nn_eigen;
                for (int i = 0; i < nn.size(); ++i) {
                    nn_eigen.emplace_back(ToVec3d(target_->points[nn[i]]));
                }

                Eigen::Vector4d n;
                if (!FitPlane(nn_eigen, n)) {
                    // 失败的不要
                    effect_pts[idx] = false;
                    return;
                }

                double dis = n.head<3>().dot(qs) + n[3];
                if (fabs(dis) > options_.max_plane_distance_) {
                    // 点离的太远了不要
                    effect_pts[idx] = false;
                    return;
                }

                effect_pts[idx] = true;

                // build residual
                Eigen::Matrix<double, 1, 6> J;
                J.block<1, 3>(0, 0) = -n.head<3>().transpose() * pose.so3().matrix() * Sophus::SO3d::hat(q);
                J.block<1, 3>(0, 3) = n.head<3>().transpose();

                jacobians[idx] = J;
                errors[idx] = dis;
            } else {
                effect_pts[idx] = false;
            }
        });

        // 累加Hessian和error,计算dx
        // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        double total_res = 0;
        int effective_num = 0;
        auto H_and_err = std::accumulate(
            index.begin(), index.end(), std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 1>::Zero()),
            [&jacobians, &errors, &effect_pts, &total_res, &effective_num](const std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>& pre,
                                                                           int idx) -> std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> {
                if (!effect_pts[idx]) {
                    return pre;
                } else {
                    total_res += errors[idx] * errors[idx];
                    effective_num++;
                    return std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(pre.first + jacobians[idx].transpose() * jacobians[idx],
                                                   pre.second - jacobians[idx].transpose() * errors[idx]);
                }
            });

        if (effective_num < options_.min_effective_pts_) {
            std::cout << "effective num too small: " << effective_num;
            return false;
        }

        Eigen::Matrix<double, 6, 6> H = H_and_err.first;
        Eigen::Matrix<double, 6, 1> err = H_and_err.second;

        Eigen::Matrix<double, 6, 1> dx = H.inverse() * err;
        pose.so3() = pose.so3() * Sophus::SO3d::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        // 更新
        std::cout << "iter " << iter << " total res: " << total_res << ", eff: " << effective_num
                  << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm() << std::endl;

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            std::cout << "iter " << iter << " pose error: " << pose_error << std::endl;
        }

        if (dx.norm() < options_.eps_) {
            std::cout << "converged, dx = " << dx.transpose() << std::endl;
            break;
        }
    }

    init_pose = pose;
    return true;
}

void Icp3d::BuildTargetKdTree() {
    kdtree_ = std::make_shared<KdTree>();
    kdtree_->BuildTree(target_);
    kdtree_->SetEnableANN();
}

bool Icp3d::AlignP2Line(Sophus::SE3d& init_pose) {
    std::cout << "aligning with point to line" << std::endl;
    assert(target_ != nullptr && source_ != nullptr);
    // 点线与点面基本是完全一样的

    Sophus::SE3d pose = init_pose;
    if (options_.use_initial_translation_) {
        pose.translation() = target_center_ - source_center_;  // 设置平移初始值
        std::cout << "init trans set to " << pose.translation().transpose() << std::endl;
    }

    std::vector<int> index(source_->points.size());
    for (int i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    std::vector<bool> effect_pts(index.size(), false);
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(index.size());
    std::vector<Eigen::Vector3d> errors(index.size());

    for (int iter = 0; iter < options_.max_iteration_; ++iter) {
        // gauss-newton 迭代
        // 最近邻，可以并发
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](int idx) {
            auto q = ToVec3d(source_->points[idx]);
            Eigen::Vector3d qs = pose * q;  // 转换之后的q
            std::vector<int> nn;
            kdtree_->GetClosestPoint(ToPointType(qs), nn, 5);  // 这里取5个最近邻
            if (nn.size() == 5) {
                // convert to eigen
                std::vector<Eigen::Vector3d> nn_eigen;
                for (int i = 0; i < 5; ++i) {
                    nn_eigen.emplace_back(ToVec3d(target_->points[nn[i]]));
                }

                Eigen::Vector3d d, p0;
                if (!FitLine(nn_eigen, p0, d, options_.max_line_distance_)) {
                    // 失败的不要
                    effect_pts[idx] = false;
                    return;
                }

                Eigen::Vector3d err = Sophus::SO3d::hat(d) * (qs - p0);

                if (err.norm() > options_.max_line_distance_) {
                    // 点离的太远了不要
                    effect_pts[idx] = false;
                    return;
                }

                effect_pts[idx] = true;

                // build residual
                Eigen::Matrix<double, 3, 6> J;
                J.block<3, 3>(0, 0) = -Sophus::SO3d::hat(d) * pose.so3().matrix() * Sophus::SO3d::hat(q);
                J.block<3, 3>(0, 3) = Sophus::SO3d::hat(d);

                jacobians[idx] = J;
                errors[idx] = err;
            } else {
                effect_pts[idx] = false;
            }
        });

        // 累加Hessian和error,计算dx
        // 原则上可以用reduce并发，写起来比较麻烦，这里写成accumulate
        double total_res = 0;
        int effective_num = 0;
        auto H_and_err = std::accumulate(
            index.begin(), index.end(), std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(Eigen::Matrix<double, 6, 6>::Zero(), Eigen::Matrix<double, 6, 1>::Zero()),
            [&jacobians, &errors, &effect_pts, &total_res, &effective_num](const std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>& pre,
                                                                           int idx) -> std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>> {
                if (!effect_pts[idx]) {
                    return pre;
                } else {
                    total_res += errors[idx].dot(errors[idx]);
                    effective_num++;
                    return std::pair<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>(pre.first + jacobians[idx].transpose() * jacobians[idx],
                                                   pre.second - jacobians[idx].transpose() * errors[idx]);
                }
            });

        if (effective_num < options_.min_effective_pts_) {
            std::cout << "effective num too small: " << effective_num;
            return false;
        }

        Eigen::Matrix<double, 6, 6> H = H_and_err.first;
        Eigen::Matrix<double, 6, 1> err = H_and_err.second;

        Eigen::Matrix<double, 6, 1> dx = H.inverse() * err;
        pose.so3() = pose.so3() * Sophus::SO3d::exp(dx.head<3>());
        pose.translation() += dx.tail<3>();

        if (gt_set_) {
            double pose_error = (gt_pose_.inverse() * pose).log().norm();
            std::cout << "iter " << iter << " pose error: " << pose_error << std::endl;
        }

        // 更新
        std::cout << "iter " << iter << " total res: " << total_res << ", eff: " << effective_num
                  << ", mean res: " << total_res / effective_num << ", dxn: " << dx.norm() << std::endl;

        if (dx.norm() < options_.eps_) {
            std::cout << "converged, dx = " << dx.transpose() << std::endl;
            break;
        }
    }

    init_pose = pose;
    return true;
}

}  // namespace sad