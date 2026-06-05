#include "pythonmanager.h"
#include <iostream>
#include <vector>

// Platform-specific headers
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

fs::path GetExecutableDir() {
#ifdef WIN32
    wchar_t buffer[MAX_PATH];
    DWORD size = GetModuleFileNameW(NULL, buffer, MAX_PATH);
    if (size == 0 || size == MAX_PATH) return "";
    return fs::path(buffer).parent_path();
#else
    char buffer[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", buffer, PATH_MAX);
    if (count == -1) return "";
    buffer[count] = '\0';
    return fs::path(buffer).parent_path();
#endif
}

PythonManager::PythonManager(int argc, char* argv[]) {

    fs::path exe_dir = GetExecutableDir();
    // assume exe is in bin directory
    this->base_path = exe_dir.parent_path();

    PyStatus status;

    // 1. Initialize the Pre-Configuration layout
    PyPreConfig preconfig;
    PyPreConfig_InitIsolatedConfig(&preconfig);

    // Turn on global UTF-8 Mode here!
    preconfig.utf8_mode = 1;

    // Commit the pre-config state to the runtime engine
    status = Py_PreInitialize(&preconfig);
    if (PyStatus_Exception(status)) {
        // Handle initialization failure
        Py_ExitStatusException(status);
    }

    PyConfig config;
    PyConfig_InitIsolatedConfig(&config);

    // 1. Set Argv
    status = PyConfig_SetBytesArgv(&config, argc, argv);
    handleStatus(status, "Failed to set Argv");

    // 2. Configure Paths
    status = PyConfig_SetString(&config, &config.home, base_path.wstring().c_str());
    handleStatus(status, "Failed to set Python Home");

    config.module_search_paths_set = 1;

    // Build our staged paths
    auto addPath = [&config](fs::path& path) {
      wchar_t* w_path = Py_DecodeLocale(path.string().c_str(), NULL);
      if (w_path != NULL) {
        PyWideStringList_Append(&config.module_search_paths, w_path);
        PyMem_RawFree(w_path);
      }
    };

    // these are order dependent
    addPath(base_path);
    fs::path stdlib_path = base_path / "lib" / "python_stdlib";
    addPath(stdlib_path);
#ifdef __linux__
    fs::path dynload_path = stdlib_path / "lib-dynload";
    addPath(dynload_path);
#endif
#ifdef _WIN32
    fs::path dlls_path = base_path / "lib" / "python_dlls";
    addPath(dlls_path);
#endif

    fs::path site_pkgs_path = base_path / "lib";
    addPath(site_pkgs_path);

    config.configure_c_stdio = 1; // Align standard I/O streams with this behavior

    // 3. Initialize Engine
    status = Py_InitializeFromConfig(&config);
    handleStatus(status, "Failed to initialize Python Engine");

    PyConfig_Clear(&config);

    // 4. Prototype Polish: Force unbuffered output
    PyRun_SimpleString("import sys; sys.stdout.reconfigure(line_buffering=True)");
}

PythonManager::~PythonManager() {
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
}



void PythonManager::handleStatus(const PyStatus& status, const std::string& msg) {
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status); // This handles the fatal exit for you
    }
}

bool PythonManager::runModule(const std::string& moduleName, const std::string& functionName) {
    PyObject* pModule = PyImport_ImportModule(moduleName.c_str());
    if (!pModule) {
        PyErr_Print();
        return false;
    }

    PyObject* pFunc = PyObject_GetAttrString(pModule, functionName.c_str());
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pValue = PyObject_CallObject(pFunc, NULL);

        if (PyErr_Occurred()) {
            if (PyErr_ExceptionMatches(PyExc_SystemExit)) {
                PyErr_Clear();
            } else {
                PyErr_Print();
            }
        }
        Py_XDECREF(pValue);
    }

    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
    return true;
}
