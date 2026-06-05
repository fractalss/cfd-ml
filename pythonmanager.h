#pragma once
#include <Python.h>
#include <string>
#include <vector>
#include <stdexcept>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

class PythonManager {
public:
    PythonManager(int argc, char* argv[]);
    ~PythonManager();

    // Prevent copying to avoid multiple initializations/finalizations
    PythonManager(const PythonManager&) = delete;
    PythonManager& operator=(const PythonManager&) = delete;

    bool runModule(const std::string& moduleName, const std::string& functionName);

private:
    void handleStatus(const PyStatus& status, const std::string& msg);
    fs::path base_path;
};
