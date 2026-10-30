/*
 * @作者：何祜宁
 * @版权所有: 智元创新（上海）科技股份有限公司
 * @日期：2025年11月21日
 * @功能：将脚踝pitch和roll与单腿J5J6位置速度力矩进行互相转换，同时增加了适配强化学习专属的转换
 */

#pragma once
#include "a3_ankle_waist_solver/solver_math_utils.hpp"

namespace zy {
// 脚踝pitch和roll的状态
using ankle_pr = vec2d;
// 脚踝J5 J6的状态
using ankle_j5j6 = vec2d;

class AnkleAnalyticalSolver {
 public:
  /*!
   * @param leg_index 0: left leg ; 1: right leg
   */
  explicit AnkleAnalyticalSolver(int leg_index);

  ~AnkleAnalyticalSolver();

  /*!
   * by hhn
   * @param pr pitch and roll configuration of ankle
   * @return motor configuration
   */
  ankle_j5j6 IK(const ankle_pr &pr) const;

  /*!
   * by hhn
   * @param mc motor configuration
   * @return pitch and roll configuration of ankle
   */
  ankle_pr FK(const ankle_j5j6 &mc) const;

  /*!
   * by hhn
   * call before UpdateJ1
   * @param vel_pr velocity of ankle pitch roll, rad/s
   * @return velocity of motors
   */
  ankle_j5j6 DIK(const ankle_pr &vel_pr) const;

  /*!
   * by hhn
   * call before UpdateJ1
   * @param vel_m velocity of motors
   * @return velocity of ankle pitch roll
   */
  ankle_pr DFK(const ankle_j5j6 &vel_m) const;

  /*!
   * by hhn
   * call before UpdateJ1
   * @param tau_pr torque of ankle pitch and roll joint
   * @return torque of motors
   */
  ankle_j5j6 IDyn(const ankle_pr &tau_pr) const;

  /*!
   * by hhn
   * call before UpdateJ1
   * @param tau_m torque of motors
   * @return torque of ankle pitch roll
   */
  ankle_pr FDyn(const ankle_j5j6 &tau_m) const;

  /*!
   * by hhn
   * call before UpdateJ1
   * @param kp kp of ankle pitch roll, should be in range (0, 500]
   * @param pr_err pitch and roll error: des_pr minus actual pr, should be in range [-15, 15]
   * @return delta_pos of motors
   */
  ankle_j5j6 RlConvertPos(const ankle_pr &kp, const ankle_pr &pr_err);

  /*!
   * by hhn
   * call before UpdateJ1
   * @param kd kd of ankle pitch roll, should be in range (0, 30]
   * @return kd of motors
   */
  ankle_j5j6 RlConvertKd(const ankle_pr &kd);

 private:
  struct Impl;
  Impl *impl_;
};
}  // namespace zy