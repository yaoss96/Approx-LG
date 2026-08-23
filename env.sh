# Put the local OSS CAD Suite (yosys, abc, iverilog) on PATH.
#
#   source env.sh
#
# Extract the suite into <repo>/oss-cad-suite; binaries are at
# https://github.com/YosysHQ/oss-cad-suite-build/releases.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) HERE_NATIVE="$(cd "$HERE" && pwd -W)" ;;
  *)                    HERE_NATIVE="$HERE" ;;
esac

export YOSYSHQ_ROOT="$HERE_NATIVE/oss-cad-suite/"
export PATH="$HERE/oss-cad-suite/bin:$HERE/oss-cad-suite/lib:$PATH"
