.PHONY: build clean

build:
	g++ -shared -fPIC -O2 -o lib/libfastcdr_compat.so lib/fastcdr_compat.cpp -L/opt/ros/$${ROS_DISTRO:-jazzy}/lib -lfastcdr -Wl,-rpath,/opt/ros/$${ROS_DISTRO:-jazzy}/lib
	colcon build --packages-select exo_head_slam --symlink-install
	@echo "Fixing Python shebangs for conda env..."
	@find install/exo_head_slam/lib/exo_head_slam -type f -executable -exec sed -i "1s|^#!.*python.*|#!$$(which python3)|" {} \;
	@echo "Build complete. Run: source install/setup.bash"

clean:
	rm -rf build install log
