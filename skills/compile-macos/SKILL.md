name: mindspore-compile-macos
description: Compile MindSpore from source on macOS (Intel/Apple Silicon), resolving CMake compatibility and network issues
user-invocable: true
tags: [mindspore, compilation, macos, cmake, build]
version: 1.0

---

TRIGGER when:
- User wants to compile MindSpore from source on macOS
- User encounters CMake version compatibility errors during MindSpore compilation
- User faces GitHub network timeout issues when building MindSpore
- User asks about MindSpore source build on Mac
- Code/logs show MindSpore compilation failures

DO NOT TRIGGER when:
- Installing MindSpore via pip/conda (use official packages instead)
- Compiling on Linux or Windows (different procedures)
- General CMake or build system questions unrelated to MindSpore
- Other deep learning frameworks (PyTorch, TensorFlow, etc.)

---

## Quick Start

```bash
# 1. Setup environment
conda create -n mindspore_py310 python=3.10 -y
conda activate mindspore_py310
conda install cmake=3.22.3 -y

# 2. Clone repository
git clone https://github.com/mindspore-ai/mindspore.git
cd mindspore
git checkout r2.8

# 3. Compile with Gitee mirror (for network issues)
bash build.sh -e cpu -j8 -S on

# 4. Install
pip install output/mindspore-*.whl
```

## Critical Requirements

**CMake Version**: Must use 3.22.3 (NOT 4.x)
- MindSpore dependencies (protobuf, c-ares) incompatible with CMake 4.x
- Fix: `conda install cmake=3.22.3 -y`

**Python Version**: 3.9.0 - 3.10.x (recommend 3.10.14)

**Network Issues**: Use `-S on` flag to enable Gitee mirror
- Resolves GitHub timeout errors
- Downloads complete in 1-3s vs 75s+ timeout

## Common Issues

### Issue 1: CMake Compatibility Error
```
CMake Error: Compatibility with CMake < 3.5 has been removed
```
**Fix**: Downgrade to CMake 3.22.3
```bash
conda install cmake=3.22.3 -y
cmake --version  # Verify
```

### Issue 2: GitHub Network Timeout
```
Failed to connect to github.com port 443 after 75023 ms
```
**Fix**: Enable Gitee mirror
```bash
bash build.sh -e cpu -j8 -S on
```

### Issue 3: sqlite Missing Gitee Support
**Symptom**: sqlite still downloads from GitHub even with `-S on`

**Fix**: Edit `cmake/external_libs/sqlite.cmake`
```cmake
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/sqlite/repository/archive/version-3.46.1.tar.gz")
    set(SHA256 "99c578c9326b12374a64dedae88a63d17557b5d2b0ac65122be67cb3fa2703da")
else()
    set(REQ_URL "https://github.com/sqlite/sqlite/archive/version-3.46.1.tar.gz")
    set(SHA256 "99c578c9326b12374a64dedae88a63d17557b5d2b0ac65122be67cb3fa2703da")
endif()
```

### Issue 4: pocketfft Missing Gitee Support
**Fix**: Use local file in `cmake/external_libs/pocketfft.cmake`
```cmake
if(ENABLE_GITEE)
    set(REQ_URL "file:///path/to/mindspore/local_deps/pocketfft-cpp.zip")
    set(SHA256 "7c475524c264c450b78e221046d90b859316e105d3d6a69d5892baeafad95493")
else()
    set(REQ_URL "https://github.com/malfet/pocketfft/archive/refs/heads/cpp.zip")
    set(SHA256 "7c475524c264c450b78e221046d90b859316e105d3d6a69d5892baeafad95493")
endif()
```

### Issue 5: Build Cache Conflicts
**Symptom**: "No rule to make target 'Makefile'"

**Fix**: Always clean build directory after cmake changes
```bash
rm -rf build/
bash build.sh -e cpu -j8 -S on
```

## Build Parameters

```bash
bash build.sh [OPTIONS]

Key options:
  -e cpu|gpu|ascend    # Target platform
  -j[n]                # Parallel threads (default 8)
  -S on|off            # Enable Gitee mirror (critical for network issues)
  -d                   # Debug mode
  -i                   # Incremental build (avoid after cmake changes)
```

## Troubleshooting Checklist

1. **Verify CMake**: `cmake --version` → Must be 3.22.3
2. **Verify Python**: `python --version` → Should be 3.10.x
3. **Check disk space**: `df -h` → Need 20GB+ free
4. **Clean build**: `rm -rf build/` before retry
5. **Check logs**: `tail -100 build.log | grep -i error`

## Best Practices

✅ DO:
- Use CMake 3.22.3 exactly
- Enable Gitee mirror with `-S on`
- Clean build directory after cmake modifications
- Verify file integrity with `shasum -a 256`

❌ DON'T:
- Use CMake 4.x
- Use incremental build (`-i`) after cmake changes
- Rely on third-party proxies (ghproxy.com)
- Skip build directory cleanup

## References

- Official docs: https://www.mindspore.cn/install
- GitHub: https://github.com/mindspore-ai/mindspore
- Gitee mirror: https://gitee.com/mindspore/mindspore
