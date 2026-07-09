#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
PREFIX_DIR="$THIRD_PARTY_DIR/install"
PANGOLIN_DIR="$THIRD_PARTY_DIR/Pangolin"
ORB_DIR="$THIRD_PARTY_DIR/ORB_SLAM3"

mkdir -p "$THIRD_PARTY_DIR" "$PREFIX_DIR"

if [[ ! -d "$PANGOLIN_DIR/.git" ]]; then
	git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git "$PANGOLIN_DIR"
fi

PANGOLIN_JPG="$PANGOLIN_DIR/src/image/image_io_jpg.cpp"
if ! grep -q '#include <cstdint>' "$PANGOLIN_JPG"; then
	sed -i '/#  include <jpeglib.h>/a #include <cstdint>' "$PANGOLIN_JPG"
fi

PANGOLIN_TAGS="$PANGOLIN_DIR/include/pangolin/log/packetstream_tags.h"
if ! grep -q '#include <cstdint>' "$PANGOLIN_TAGS"; then
	sed -i '1a #include <cstdint>' "$PANGOLIN_TAGS"
fi

PANGOLIN_COLOUR="$PANGOLIN_DIR/include/pangolin/gl/colour.h"
if ! grep -q '#include <limits>' "$PANGOLIN_COLOUR"; then
	sed -i '/#include <cmath>/a #include <limits>' "$PANGOLIN_COLOUR"
fi

cmake -S "$PANGOLIN_DIR" -B "$PANGOLIN_DIR/build" \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX="$PREFIX_DIR" \
	-DBUILD_PANGOLIN_PYTHON=OFF \
	-DBUILD_PYPANGOLIN_MODULE=OFF \
	-DBUILD_TESTS=OFF \
	-DBUILD_TOOLS=OFF \
	-DBUILD_EXAMPLES=OFF
cmake --build "$PANGOLIN_DIR/build" -j"$(nproc)"
cmake --install "$PANGOLIN_DIR/build"

if [[ ! -d "$ORB_DIR/.git" ]]; then
	git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$ORB_DIR"
fi

export CMAKE_PREFIX_PATH="$PREFIX_DIR:${CMAKE_PREFIX_PATH:-}"
cd "$ORB_DIR"
if [[ -f Vocabulary/ORBvoc.txt.tar.gz && ! -f Vocabulary/ORBvoc.txt ]]; then
	tar -xzf Vocabulary/ORBvoc.txt.tar.gz -C Vocabulary
fi
chmod +x build.sh
./build.sh

cat <<EOF
ORB-SLAM3 built.

Run before rebuilding this ROS workspace:
  export ORB_SLAM3_ROOT="$ORB_DIR"
  export CMAKE_PREFIX_PATH="$PREFIX_DIR:\${CMAKE_PREFIX_PATH:-}"
  make build
EOF
