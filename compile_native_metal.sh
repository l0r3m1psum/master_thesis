# https://keith.github.io/xcode-man-pages/
# Creates a default.metallib
xcrun -sdk macosx metal -arch amdgpu_gfx803 -N test.mtlp-json ../kernels.metal
# Saddly we can't do -d
xcrun -sdk macosx metal-objdump -s default.metallib
# /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/metal/macos/
