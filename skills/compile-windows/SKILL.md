name: mindspore-compile-windows
description: Compile MindSpore from source on Windows x86-64, with Visual Studio, CMake, and MSYS2 setup
user-invocable: true
tags: [mindspore, compilation, windows, visual-studio, cmake, msys2]
version: 1.0

---

TRIGGER when:
- User wants to compile MindSpore from source on Windows
- User encounters Visual Studio or MSVC compilation errors for MindSpore
- User faces CMake or build tool issues on Windows during MindSpore build
- User asks about MindSpore Windows source compilation
- Code/logs show MindSpore Windows build failures

DO NOT TRIGGER when:
- Installing MindSpore via pip (use official packages instead)
- Compiling on macOS or Linux (different procedures)
- General Windows development questions unrelated to MindSpore
- Other deep learning frameworks (PyTorch, TensorFlow, etc.)

---

## Quick Start

```cmd
REM 1. Setup environment (run as Administrator)
call "C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\msys64\usr\bin;C:\Python310;C:\Python310\Scripts;C:\Program Files\CMake\bin;%PATH%

REM 2. Clone repository
cd C:\mindspore_build
git clone https://gitee.com/mindspore/mindspore.git
cd mindspore
git checkout r2.8

REM 3. Compile
call build.bat -e cpu -j 8

REM 4. Install
pip install output\mindspore-*.whl
```

## System Requirements

| Software | Version | Critical Notes |
|----------|---------|----------------|
| Windows | 10/11 x64 | Must be 64-bit |
| Visual Studio | latest Community | With C++ workload |
| CMake | **3.22.3** | NOT 3.23+ or 4.x |
| Python | 3.9.0 - 3.10.x | Recommend 3.10.14 |
| MSYS2 | Latest | Provides Unix tools |
| Git | Latest | Add to PATH |

## Critical Setup Steps

### 1. Install latest Visual Studio
**Required workloads**:
- ✅ Desktop development with C++
- ✅ Universal Windows Platform development
- ✅ C++ CMake tools for Windows

### 2. Install CMake 3.22.3 (EXACT VERSION)
**Download**: https://cmake.org/files/v3.22/cmake-3.22.3-windows-x86_64.msi

**Critical**: Must use 3.22.3, NOT 3.23+ or 4.x
- Install to: `C:\Program Files\CMake`
- Select: "Add CMake to system PATH for all users"
- Verify: `cmake --version` → Should show 3.22.3

### 3. Install MSYS2
**Download**: https://www.msys2.org/

**Setup**:
```bash
# In MSYS2 terminal
pacman -Syu
# Close and reopen terminal
pacman -Su
pacman -S patch make
```

**Add to PATH** (must be FIRST):
```
C:\msys64\usr\bin
```

### 4. Install Python 3.10
**Install packages**:
```cmd
python -m pip install --upgrade pip
pip install wheel>=0.32.0
pip install "pyyaml>=6.0,<=6.0.2"
pip install "numpy>=1.19.3,<=1.26.4"
```

## Environment Setup Script

Create `build_env.bat`:
```batch
@echo off
REM Load Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Set paths (MSYS2 must be first)
set PATH=C:\msys64\usr\bin;C:\Python310;C:\Python310\Scripts;C:\Program Files\CMake\bin;%PATH%

REM Verify
echo Environment ready!
python --version
cmake --version
patch --version
```

## Common Issues

### Issue 1: CMake Version Incompatible
```
CMake Error: Compatibility with CMake < 3.5 has been removed
```
**Fix**: Uninstall current CMake, install 3.22.3 exactly
```cmd
cmake --version  # Must show 3.22.3
```

### Issue 2: 'patch' Not Recognized
```
'patch' is not recognized as an internal or external command
```
**Fix**: Ensure MSYS2 in PATH (must be FIRST)
```cmd
set PATH=C:\msys64\usr\bin;%PATH%
patch --version
```

### Issue 3: Visual Studio Compiler Not Found
```
error: Microsoft Visual C++ 14.0 or greater is required
```
**Fix**: Load VS environment
```cmd
call "C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Auxiliary\Build\vcvars64.bat"
cl  # Should show MSVC version
```

### Issue 4: Network Download Timeout
**Symptom**: Failed to download packages from GitHub

**Solutions**:
A. Use proxy:
```cmd
set HTTP_PROXY=http://proxy.example.com:8080
set HTTPS_PROXY=http://proxy.example.com:8080
```

B. Modify cmake files to use mirrors (similar to macOS version)

C. Manual download and place in cache directory

### Issue 5: Path Contains Spaces or Chinese
```
CMake Error: The source directory contains spaces or non-ASCII characters
```
**Fix**: Move to pure English path
```cmd
move "C:\用户\文档\mindspore" C:\mindspore_build\mindspore
```

### Issue 6: Disk Space Insufficient
**Requirement**: At least 30GB free space
```cmd
dir C:\  # Check available space
rmdir /s /q build  # Clean if needed
```

## Build Parameters

```cmd
call build.bat [OPTIONS]

Options:
  -e cpu     # CPU version
  -e gpu     # GPU version (requires CUDA)
  -j 8       # Use 8 parallel threads
```

## Windows-Specific Considerations

1. **Path Separators**: Use `\` or `/` (escape `\` in scripts)
2. **Administrator Rights**: Run Command Prompt as Administrator
3. **Antivirus**: Add build directory to whitelist
4. **Long Path Support**: Enable if needed
```cmd
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```
5. **File Encoding**: Use UTF-8, avoid Notepad (use VSCode)

## PATH Configuration Order (CRITICAL)

```
C:\msys64\usr\bin                    # MSYS2 (MUST BE FIRST)
C:\Python310                         # Python
C:\Python310\Scripts                 # Python scripts
C:\Program Files\CMake\bin           # CMake
D:\Program Files\Git\usr\bin         # Git Unix tools
D:\Program Files\Git\bin             # Git
%SystemRoot%\system32                # Windows
```

**Configure**: Right-click "This PC" → Properties → Advanced → Environment Variables → System Variables → Path

## Complete Build Script

Create `compile_mindspore.bat`:
```batch
@echo off
echo ========================================
echo MindSpore Windows Compilation
echo ========================================

REM 1. Setup environment
echo [1/5] Setting up environment...
call "C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\msys64\usr\bin;C:\Python310;C:\Python310\Scripts;C:\Program Files\CMake\bin;%PATH%

REM 2. Verify
echo [2/5] Verifying environment...
python --version
cmake --version
if errorlevel 1 (
    echo ERROR: Environment verification failed!
    pause
    exit /b 1
)

REM 3. Clean
echo [3/5] Cleaning old build...
if exist build rmdir /s /q build

REM 4. Compile
echo [4/5] Starting compilation...
call build.bat -e cpu -j 8
if errorlevel 1 (
    echo ERROR: Compilation failed!
    pause
    exit /b 1
)

REM 5. Install
echo [5/5] Installing...
for %%f in (output\mindspore-*.whl) do pip install %%f

echo ========================================
echo Compilation completed successfully!
echo ========================================
pause
```

## Troubleshooting Checklist

1. **Check PATH order**:
```cmd
echo %PATH%
# Verify MSYS2 is first
```

2. **Check versions**:
```cmd
cmake --version        # Must be 3.22.3
python --version       # Should be 3.10.x
cl                     # Should show MSVC
patch --version        # Should work
```

3. **Check Python packages**:
```cmd
pip list | findstr wheel
pip list | findstr pyyaml
pip list | findstr numpy
```

4. **Check disk space**: Need 30GB+ free

5. **Check logs**:
```cmd
type build\mindspore\CMakeFiles\CMakeError.log
```

6. **Clean and retry**:
```cmd
rmdir /s /q build
call build_env.bat
call build.bat -e cpu -j 8 > build.log 2>&1
```

## Performance Optimization

1. **Use SSD**: Place source and build on SSD
2. **Virtual Memory**: Set to 1.5-2x physical RAM
3. **Parallel Threads**: Match CPU cores
```cmd
call build.bat -e cpu -j 16  # For 16-core CPU
```
4. **Close Background Apps**: Free up resources during build

## Key Differences from macOS

| Aspect | macOS | Windows |
|--------|-------|---------|
| Compiler | Clang/GCC | MSVC 2026 |
| Package Manager | conda/brew | pip/chocolatey |
| Shell | bash/zsh | cmd/PowerShell |
| Path Separator | `/` | `\` |
| Unix Tools | Native | Requires MSYS2 |
| Build Time | 1-2 hours | 1-3 hours |

## References

- Official docs: https://www.mindspore.cn/install
- Windows guide: https://www.mindspore.cn/install/detail?path=install/r2.3/mindspore_cpu_win_install_source.md
- GitHub: https://github.com/mindspore-ai/mindspore
- Gitee mirror: https://gitee.com/mindspore/mindspore
