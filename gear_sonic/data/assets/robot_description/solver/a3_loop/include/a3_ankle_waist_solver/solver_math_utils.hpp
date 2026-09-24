/*
 * @作者：何祜宁
 * @版权所有: 智元创新（上海）科技股份有限公司
 * @日期：2025年11月21日
 * @功能：脚踝和腰部解算依赖的矩阵计算，避免引入其他三方库
 */

#pragma once
#include <cmath>
#include <iostream>
#include <array>

namespace zy {
    // 二维度向量与矩阵
    using vec2d = std::array<double, 2>;
    using mat22d = std::array<std::array<double, 2>, 2>;

    // 2x2 矩阵与 2x2 矩阵相乘
    inline mat22d mat22dMulmat22d(const mat22d &m1, const mat22d &m2) {
        mat22d result;
        result[0][0] = m1[0][0] * m2[0][0] + m1[0][1] * m2[1][0];
        result[0][1] = m1[0][0] * m2[0][1] + m1[0][1] * m2[1][1];
        result[1][0] = m1[1][0] * m2[0][0] + m1[1][1] * m2[1][0];
        result[1][1] = m1[1][0] * m2[0][1] + m1[1][1] * m2[1][1];
        return result;
    }

    // 2x2 矩阵与 2x1 向量相乘
    inline vec2d mat22dProduct(const mat22d &m1, const vec2d& v1) {
        vec2d result;
        result[0] = m1[0][0] * v1[0] + m1[0][1] * v1[1];
        result[1] = m1[1][0] * v1[0] + m1[1][1] * v1[1];
        return result;
    }

    // 计算2x2矩阵的行列式
    inline double mat22dDeterminant(const mat22d &m) {
        return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    }

    // 求2x2矩阵的逆矩阵
    inline mat22d mat22dInverse(const mat22d &m) {
        double det = mat22dDeterminant(m);

        // 检查矩阵是否可逆
        if (std::abs(det) < 1e-6) {
            det = 1e-6;
            // std::cerr << "Matrix is singular (non-invertible)" << std::endl;
        }

        double inv_det = 1.0 / det;
        mat22d result;

        // 2x2矩阵逆的公式：
        // [a b]^-1   = 1/(ad-bc) * [d -b]
        // [c d]                   [-c a]
        result[0][0] = m[1][1] * inv_det;
        result[0][1] = -m[0][1] * inv_det;
        result[1][0] = -m[1][0] * inv_det;
        result[1][1] = m[0][0] * inv_det;

        return result;
    }

    // 求2x2矩阵的转置
    inline mat22d mat22dTranspose(const mat22d &m) {
        mat22d result;

        // 转置操作：行列互换
        result[0][0] = m[0][0]; // 左上角不变
        result[0][1] = m[1][0]; // 第一行第二列 = 第二行第一列
        result[1][0] = m[0][1]; // 第二行第一列 = 第一行第二列
        result[1][1] = m[1][1]; // 右下角不变

        return result;
    }
}

inline std::ostream &operator<<(std::ostream &os, const zy::mat22d &m) {
    os << "[";
    for (size_t i = 0; i < m.size(); i++) {
        if (i > 0) os << " ";
        os << "[";
        for (size_t j = 0; j < m[i].size(); j++) {
            os << m[i][j];
            if (j < m[i].size() - 1) os << ", ";
        }
        os << "]";
        if (i < m.size() - 1) os << "\n";
    }
    os << "]";
    return os;
}

inline std::ostream &operator<<(std::ostream &os, const zy::vec2d &m) {
    os << "[";
    for (size_t i = 0; i < m.size(); i++) {
        os << m[i];
        if (i < m.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}
