/*
 * @作者：何祜宁
 * @版权所有: 智元创新（上海）科技股份有限公司
 * @日期：2025年11月21日
 * @功能：将腰部pitch和roll与与两个关节位置速度力矩进行互相转换，同时增加了适配强化学习专属的转换
 */

#pragma once
#include "a3_ankle_waist_solver/solver_math_utils.hpp"

namespace zy {
// 脚踝pitch和roll的状态
using waist_pr = vec2d;
// 脚踝J5 J6的状态
using waist_j1j2 = vec2d;

class WaistAnalyticalSolver {
 public:
  /*!
   * @param leg_index 0: left leg ; 1: right leg
   */
  WaistAnalyticalSolver();

  ~WaistAnalyticalSolver();

  /*!
   * by hhn
   * @param pr pitch and roll configuration of waist
   * @return motor configuration
   */
  waist_j1j2 IK(const waist_pr &pr) const;

  /*!
   * by hhn
   * @param mc motor configuration
   * @return pitch and roll configuration of waist
   */
  waist_pr FK(const waist_j1j2 &mc) const;

  /*!
   * by hhn
   * call before UpdateJ1
   * @param vel_pr velocity of waist pitch roll, rad/s
   * @return velocity of motors
   */
  waist_j1j2 DIK(const waist_pr &vel_pr) const;

  /*!
   * by hhn
   * call before UpdateJ1
   * @param vel_m velocity of motors
   * @return velocity of waist pitch roll
   */
  waist_pr DFK(const waist_j1j2 &vel_m) const;

  /*!
   * by hhn
   * call before UpdateJ1
   * @param tau_pr torque of waist pitch and roll joint
   * @return torque of motors
   */
  waist_j1j2 IDyn(const waist_pr &tau_pr) const;

  /*!
   * by hhn
   * call before UpdateJ1
   * @param tau_m torque of motors
   * @return torque of waist pitch roll
   */
  waist_pr FDyn(const waist_j1j2 &tau_m) const;

  /*!
   * by hhn
   * call before UpdateJ1
   * @param kp kp of waist pitch roll, should be in range (0, 500]
   * @param pr_err pitch and roll error: des_pr minus actual pr, should be in range [-15, 15]
   * @return delta_pos of motors
   */
  waist_j1j2 RlConvertPos(const waist_pr &kp, const waist_pr &pr_err);

  /*!
   * by hhn
   * call before UpdateJ1
   * @param kd kd of waist pitch roll, should be in range (0, 30]
   * @return kd of motors
   */
  waist_j1j2 RlConvertKd(const waist_pr &kd);

 private:
  struct Impl;
  Impl *impl_;
};
}  // namespace zy