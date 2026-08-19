#!/usr/bin/env bash
# 解决 LightGBM 在 macOS 缺少 libomp 的问题（可复现）。
# 优先用 brew 安装的 libomp；若没有，则复用同环境 scikit-learn 自带的 libomp.dylib。
set -e
VENV_LIB=$(python -c "import lightgbm, os; print(os.path.dirname(lightgbm.__file__))")/lib
LGB_DYLIB="$VENV_LIB/lib_lightgbm.dylib"

# 1) 尝试系统/brew 的 libomp
if [ -f /opt/homebrew/opt/libomp/lib/libomp.dylib ]; then
  cp /opt/homebrew/opt/libomp/lib/libomp.dylib "$VENV_LIB/libomp.dylib"
elif [ -f /usr/local/opt/libomp/lib/libomp.dylib ]; then
  cp /usr/local/opt/libomp/lib/libomp.dylib "$VENV_LIB/libomp.dylib"
else
  # 2) 复用 sklearn 自带
  SKLEARN_OMP=$(python -c "import sklearn, glob, os; print(glob.glob(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs','libomp.dylib'))[0])")
  cp "$SKLEARN_OMP" "$VENV_LIB/libomp.dylib"
fi

# 3) 给 lightgbm dylib 添加 @loader_path rpath，使其在同目录找到 libomp
install_name_tool -add_rpath @loader_path "$LGB_DYLIB" 2>/dev/null || true
python -c "import lightgbm; print('lightgbm', lightgbm.__version__, 'OK')"
