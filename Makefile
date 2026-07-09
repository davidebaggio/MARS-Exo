.PHONY: build clean

build:
	colcon build --paths . ros2_wrappers/orbslam3_ros2 --packages-up-to exo_head_slam --symlink-install
	@echo "Fixing Python shebangs for conda env..."
	@find install/exo_head_slam/lib/exo_head_slam -type f -executable -exec sed -i "1s|^#!.*python.*|#!$$(which python3)|" {} \;
	@if [ -d install/orbslam3_ros2/lib/orbslam3_ros2 ]; then find install/orbslam3_ros2/lib/orbslam3_ros2 -type f -executable -exec sed -i "1s|^#!.*python.*|#!$$(which python3)|" {} \;; fi
	@echo "Build complete. Run: source install/setup.bash"

clean:
	rm -rf build install log
