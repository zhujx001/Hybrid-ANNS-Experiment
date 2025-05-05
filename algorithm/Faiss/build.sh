# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Clean build directory
echo "Cleaning build directory..."
rm -rf *

# Configure project
echo "Configuring project..."
cmake ..

# Build project
echo "Building project..."
make -j$(nproc)

echo "Build completed!"
echo "Executables are located in: $BUILD_DIR/bin/ directory"