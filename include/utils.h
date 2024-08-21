#pragma once

#include <chrono>
#include <limits>
#include <numeric>


namespace sad
{

    /**
     * 计算一个容器内数据的均值与对角形式协方差
     * @tparam C    容器类型
     * @tparam D    结果类型
     * @tparam Getter   获取数据函数, 接收一个容器内数据类型，返回一个D类型
     */
    template <typename C, typename D, typename Getter>
    void ComputeMeanAndCovDiag(const C &data, D &mean, D &cov_diag, Getter &&getter)
    {
        size_t len = data.size();
        assert(len > 1);
        // clang-format off
    mean = std::accumulate(data.begin(), data.end(), D::Zero().eval(),
                           [&getter](const D& sum, const auto& data) -> D { return sum + getter(data); }) / len;
    cov_diag = std::accumulate(data.begin(), data.end(), D::Zero().eval(),
                               [&mean, &getter](const D& sum, const auto& data) -> D {
                                   return sum + (getter(data) - mean).cwiseAbs2().eval();
                               }) / (len - 1);
        // clang-format on
    }

    /**
     * 计算一个容器内数据的均值与矩阵形式协方差
     * @tparam C    容器类型
     * @tparam int 　数据维度
     * @tparam Getter   获取数据函数, 接收一个容器内数据类型，返回一个Eigen::Matrix<double, dim,1> 矢量类型
     */
    template <typename C, int dim, typename Getter>
    void ComputeMeanAndCov(const C &data, Eigen::Matrix<double, dim, 1> &mean, Eigen::Matrix<double, dim, dim> &cov,
                           Getter &&getter)
    {
        using D = Eigen::Matrix<double, dim, 1>;
        using E = Eigen::Matrix<double, dim, dim>;
        size_t len = data.size();
        assert(len > 1);

        // clang-format off
    mean = std::accumulate(data.begin(), data.end(), Eigen::Matrix<double, dim, 1>::Zero().eval(),
                           [&getter](const D& sum, const auto& data) -> D { return sum + getter(data); }) / len;
    cov = std::accumulate(data.begin(), data.end(), E::Zero().eval(),
                          [&mean, &getter](const E& sum, const auto& data) -> E {
                              D v = getter(data) - mean;
                              return sum + v * v.transpose();
                          }) / (len - 1);
        // clang-format on
    }

    template <typename S>
    bool FitPlane(std::vector<Eigen::Matrix<S, 3, 1>> &data, Eigen::Matrix<S, 4, 1> &plane_coeffs, double eps = 1e-2)
    {
        if (data.size() < 3)
        {
            return false;
        }

        Eigen::MatrixXd A(data.size(), 4);
        for (int i = 0; i < data.size(); ++i)
        {
            A.row(i).head<3>() = data[i].transpose();
            A.row(i)[3] = 1.0;
        }

        Eigen::JacobiSVD svd(A, Eigen::ComputeThinV);
        plane_coeffs = svd.matrixV().col(3);

        // check error eps
        for (int i = 0; i < data.size(); ++i)
        {
            double err = plane_coeffs.template head<3>().dot(data[i]) + plane_coeffs[3];
            if (err * err > eps)
            {
                return false;
            }
        }

        return true;
    }

    template <typename S>
    bool FitLine(std::vector<Eigen::Matrix<S, 3, 1>> &data, Eigen::Matrix<S, 3, 1> &origin, Eigen::Matrix<S, 3, 1> &dir,
                 double eps = 0.2)
    {
        if (data.size() < 2)
        {
            return false;
        }

        origin = std::accumulate(data.begin(), data.end(), Eigen::Matrix<S, 3, 1>::Zero().eval()) / data.size();

        Eigen::MatrixXd Y(data.size(), 3);
        for (int i = 0; i < data.size(); ++i)
        {
            Y.row(i) = (data[i] - origin).transpose();
        }

        Eigen::JacobiSVD svd(Y, Eigen::ComputeFullV);
        dir = svd.matrixV().col(0);

        // check eps
        for (const auto &d : data)
        {
            if (dir.template cross(d - origin).template squaredNorm() > eps)
            {
                return false;
            }
        }

        return true;
    }

    /**
    * 统计代码运行时间
    * @tparam FuncT
    * @param func  被调用函数
    * @param func_name 函数名
    * @param times 调用次数
    */
    template <typename FuncT>
    void evaluate_and_call(FuncT&& func, const std::string& func_name = "", int times = 10) {
        double total_time = 0;
        for (int i = 0; i < times; ++i) {
            auto t1 = std::chrono::high_resolution_clock::now();
            func();
            auto t2 = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
        }
        std::cout << "方法 " << func_name << " 平均调用时间/次数: " << total_time / times << "/" << times << " 毫秒." << std::endl;
    }

} // namespace sad
