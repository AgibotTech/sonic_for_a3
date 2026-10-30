# Copyright (c) 2023, AgiBot Inc.
# All rights reserved

import tkinter as tk
from tkinter import ttk
import json
import subprocess


class PIDTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("Angle PID Parameter Tuner")
        self.api_url = "http://127.0.0.1:50080/rpc/aimrt_mujoco_sim.protocols.examples.inverted_pendulum.PidControl"

        self.real_time_mode = tk.BooleanVar(value=False)

        self.default_values = {
            "p": 0.0,
            "i": 0.0,
            "d": 0.0,
        }

        # create GUI elements
        self.create_url_config()
        self.create_send_mode_control()
        self.create_step_control()
        self.create_pid_frame("Angle PID", 3)
        self.create_control_buttons()

    def create_url_config(self):
        """create API configuration frame"""
        url_frame = ttk.LabelFrame(self.root, text="API Configuration", padding="5 5 5 5")
        url_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        ttk.Label(url_frame, text="API URL:").grid(row=0, column=0, padx=5)
        self.url_var = tk.StringVar(value=self.api_url)
        url_entry = ttk.Entry(url_frame, textvariable=self.url_var, width=40)
        url_entry.grid(row=0, column=1, padx=5)

    def create_send_mode_control(self):
        """create send mode control"""
        mode_frame = ttk.LabelFrame(self.root, text="Send Mode", padding="5 5 5 5")
        mode_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        ttk.Radiobutton(mode_frame, text="Manual Send (Apply)", variable=self.real_time_mode, value=False).grid(
            row=0, column=0, padx=10
        )

        ttk.Radiobutton(mode_frame, text="Real-time Send", variable=self.real_time_mode, value=True).grid(
            row=0, column=1, padx=10
        )

    def create_step_control(self):
        """create step size control"""
        step_frame = ttk.LabelFrame(self.root, text="Step Size Control", padding="5 5 5 5")
        step_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        left_frame = ttk.Frame(step_frame)
        left_frame.grid(row=0, column=0, padx=5)

        ttk.Label(left_frame, text="Step Size:").grid(row=0, column=0, padx=5)
        self.step_var = tk.StringVar(value="0.100")
        step_entry = ttk.Entry(left_frame, textvariable=self.step_var, width=10)
        step_entry.grid(row=0, column=1, padx=5)
        step_entry.bind("<Return>", lambda e: self.validate_step_size())

        right_frame = ttk.Frame(step_frame)
        right_frame.grid(row=0, column=1, padx=5)

        quick_steps = [("0.001", 0), ("0.01", 1), ("0.1", 2), ("1.0", 3)]

        for value, col in quick_steps:
            btn = ttk.Button(right_frame, text=value, width=6, command=lambda v=value: self.set_step_size(v))
            btn.grid(row=0, column=col, padx=2)

    def create_pid_frame(self, title, row_position):
        """create PID parameter frame"""
        frame = ttk.LabelFrame(self.root, text=title, padding="5 5 5 5")
        frame.grid(row=row_position, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        params = ["Kp", "Ki", "Kd"]
        self.pid_vars = {}
        param_keys = ["p", "i", "d"]

        for i, (param, key) in enumerate(zip(params, param_keys)):
            ttk.Label(frame, text=param).grid(row=0, column=i * 3, padx=5)

            var = tk.StringVar(value=str(self.default_values[key]))
            self.pid_vars[key] = var

            entry = ttk.Entry(frame, textvariable=var, width=10)
            entry.grid(row=0, column=i * 3 + 1)
            entry.bind("<Return>", lambda e, v=var: self.on_entry_return(v))

            btn_frame = ttk.Frame(frame)
            btn_frame.grid(row=0, column=i * 3 + 2)

            ttk.Button(btn_frame, text="+", width=2, command=lambda v=var: self.adjust_value(v, 1)).grid(
                row=0, column=0
            )

            ttk.Button(btn_frame, text="-", width=2, command=lambda v=var: self.adjust_value(v, -1)).grid(
                row=1, column=0
            )

    def create_control_buttons(self):
        """create control buttons"""
        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=5, column=0, columnspan=3, pady=15)

        ttk.Button(control_frame, text="Apply Parameters", command=self.apply_parameters).grid(row=0, column=0, padx=5)

        ttk.Button(control_frame, text="Reset to Default", command=self.reset_to_default).grid(row=0, column=1, padx=5)

        ttk.Button(control_frame, text="Get Current Parameters", command=self.get_current_parameters).grid(
            row=0, column=2, padx=5
        )

    def set_step_size(self, value):
        """set step size"""
        try:
            float_val = float(value)
            self.step_var.set(f"{float_val:.5f}")
        except ValueError:
            self.step_var.set("0.100")

    def validate_step_size(self):
        """validate step size"""
        try:
            value = float(self.step_var.get())
            if value <= 0:
                raise ValueError
            self.step_var.set(f"{value:.5f}")
        except ValueError:
            self.step_var.set("0.100")

    def get_step_size(self):
        """get step size"""
        try:
            return float(self.step_var.get())
        except ValueError:
            self.step_var.set("0.100")
            return 0.1

    def adjust_value(self, var, direction):
        """adjust value"""
        try:
            current = float(var.get())
            step = self.get_step_size()
            new_value = round(current + direction * step, 6)
            if new_value >= 0:
                var.set(f"{new_value:.5f}")
                if self.real_time_mode.get():
                    self.apply_parameters()
        except ValueError:
            var.set("0.000")

    def on_entry_return(self, var):
        """handle entry return event"""
        try:
            value = float(var.get())
            var.set(f"{value:.5f}")
            if self.real_time_mode.get():
                self.apply_parameters()
        except ValueError:
            var.set("0.000")

    def apply_parameters(self):
        """apply parameters and send"""
        params = {"value": {}}
        for key, var in self.pid_vars.items():
            try:
                value = float(var.get())
                var.set(f"{value:.5f}")
                params["value"][key] = value
            except ValueError:
                var.set("0.000")
                params["value"][key] = 0.0

        print("\nApplied Parameters:")
        print(json.dumps(params, indent=2))

        try:
            curl_cmd = [
                "curl",
                "-X",
                "POST",
                "-H",
                "content-type:application/json",
                self.url_var.get() + "/SetPid",
                "-d",
                json.dumps(params),
            ]

            result = subprocess.run(curl_cmd, capture_output=True, text=True)
            print(f"Response:\n{result.stdout}")

        except Exception as e:
            print(f"Error sending parameters: {str(e)}")

    def get_current_parameters(self):
        """获取当前PID参数"""
        try:
            # 构造空的请求体
            request_data = {}

            curl_cmd = [
                "curl",
                "-X",
                "POST",
                "-H",
                "content-type:application/json",
                self.url_var.get() + "/GetPid",
                "-d",
                json.dumps(request_data),
            ]

            result = subprocess.run(curl_cmd, capture_output=True, text=True)
            print(f"Response:\n{result.stdout}")

            # 解析返回的JSON数据
            try:
                response_data = json.loads(result.stdout)
                # 根据proto格式检查返回值
                if response_data.get("code") == "0":  # 成功
                    pid_values = response_data.get("value", {})
                    if pid_values:
                        # 直接更新界面上的PID值
                        self.pid_vars["p"].set(f"{float(pid_values.get('p', 0)):.5f}")
                        self.pid_vars["i"].set(f"{float(pid_values.get('i', 0)):.5f}")
                        self.pid_vars["d"].set(f"{float(pid_values.get('d', 0)):.5f}")
                        print("Successfully updated PID values")
                else:
                    print(f"Error: {response_data.get('msg', 'Unknown error')}")

            except json.JSONDecodeError as e:
                print(f"Error parsing response: {str(e)}")

        except Exception as e:
            print(f"Error getting parameters: {str(e)}")
            print(f"Error getting parameters: {str(e)}")

    def reset_to_default(self):
        """reset to default values"""
        for key, value in self.default_values.items():
            self.pid_vars[key].set(f"{value:.5f}")
        self.apply_parameters()


def main():
    root = tk.Tk()
    app = PIDTuner(root)
    root.mainloop()


if __name__ == "__main__":
    main()
