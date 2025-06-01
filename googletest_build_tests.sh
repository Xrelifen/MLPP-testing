#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# 1. 清理 / 创造编译目录
OBJDIR=googletest_build/obj
LIB=googletest_build/libmlpp.a
rm -rf googletest_build && mkdir -p "$OBJDIR"

# 2. 编译所有 MLPP 源码为目标档
for src in $(find MLPP -type f -name '*.cpp'); do
    obj="$OBJDIR/$(basename "${src%.cpp}.o")"
    g++ -std=c++17 -I MLPP -pthread -c "$src" -o "$obj"
done

# 3. 打包成静态函式库
ar rcs "$LIB" "$OBJDIR"/*.o

# 4. 编译 GoogleTest
mkdir -p googletest/build
pushd googletest/build >/dev/null
cmake .. && make
popd >/dev/null

# 5. 编译并连结所有单元测试
g++ -std=c++17 \
    -I MLPP \
    -I googletest/googletest/include \
    -I googletest_unittest \
    -pthread \
    googletest_unittest/test_*.cpp \
    "$LIB" \
    googletest/build/lib/libgtest.a \
    googletest/build/lib/libgtest_main.a \
    -o runGoogleTests

# 6. 执行测试
./runGoogleTests
