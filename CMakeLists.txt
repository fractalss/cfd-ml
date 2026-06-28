cmake_minimum_required(VERSION 3.18)

# --- 1. CORE DETECTION ---
include(ProcessorCount)
ProcessorCount(N_CORES)
if(N_CORES EQUAL 0)
    set(N_CORES 4)
endif()

# setting up python is going to be a pain:

# =================================================================
# PASS 1: Find Host Python & Bootstrapping
# =================================================================
find_package(Python3 3.12...<3.13 REQUIRED COMPONENTS Development Interpreter)

# store this for getting dlls/dynlib later
if(WIN32)
    execute_process(
            COMMAND python -c "import sys, os; print(os.path.abspath(os.path.join(sys.prefix, 'DLLs')))"
            OUTPUT_VARIABLE REAL_SYSTEM_DLLS
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    file(TO_CMAKE_PATH "${REAL_SYSTEM_DLLS}" REAL_SYSTEM_DLLS)
else()
    get_filename_component(REAL_SYSTEM_STDLIB "${Python3_STDLIB}" REALPATH)
    file(TO_CMAKE_PATH "${REAL_SYSTEM_STDLIB}" REAL_SYSTEM_STDLIB)
endif()


set(VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/venv")

if(NOT EXISTS "${VENV_DIR}")
    message(STATUS "Creating fresh virtual environment at ${VENV_DIR}...")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m venv "${VENV_DIR}"
        RESULT_VARIABLE VENV_RETS
    )
endif()

# =================================================================
# THE RESET: Wipe CMake's Memory Cache
# =================================================================
unset(Python3_FOUND CACHE)
unset(Python3_EXECUTABLE CACHE)
unset(Python3_PREFIX CACHE)
unset(Python3_SITELIB CACHE)
unset(Python3_STDLIB CACHE)

# =================================================================
# PASS 2: Lock onto the VENV Explicitly
# =================================================================
# Note on requirements files:
# these are generated on linux and windows from the build image using the following steps
# from a master requirements.in file
# 1.  Activate the venv in src/bvr-rom (source ./venv/bin/activate  or ./venv/Scripts/Activate.bat)
# 2.  Install pip-tools if not present (pip install pip-tools)
# 3.  pip-compile --upgrade requirements.in
# 4.  copy the generated requirements.txt to either requirements-linux.txt or requirements-windows.txt

if(WIN32)
    set(Python3_EXECUTABLE "${VENV_DIR}/Scripts/python.exe")
    set(REQUIREMENTS_FILE "requirements-windows.txt")
else()
    set(Python3_EXECUTABLE "${VENV_DIR}/bin/python")
    set(REQUIREMENTS_FILE "requirements-linux.txt")
endif()

set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
cpfd_add_to_inuse_manifest(Python ${Python3_VERSION})

# =================================================================
# RIGHT HERE: Run Pip Install Requirements
# =================================================================
# Now that we know Python3_EXECUTABLE is 100% pointing to the venv,
# we use it to safely install our packages.
message(STATUS "Installing Python packages from ${REQUIREMENTS_FILE}...")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pip install --upgrade pip setuptools wheel
    RESULT_VARIABLE PIP_RET
)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
    "${Python3_EXECUTABLE}" -m pip install -r "${CMAKE_CURRENT_SOURCE_DIR}/${REQUIREMENTS_FILE}"
    RESULT_VARIABLE PIP_RET
)

if(NOT PIP_RET EQUAL 0)
    message(FATAL_ERROR "Pip requirements installation failed!")
endif()

# =================================================================
# PASS 3: File Math & Packager Loops
# =================================================================
# Since requirements are now physically sitting on disk in the venv,
# this path math will successfully find things like numpy and torch!
get_filename_component(ABS_PYTHON_SITELIB "${Python3_SITELIB}" ABSOLUTE)
if(WIN32)
    # On Windows, site-packages is explicitly nested under Lib/site-packages
    set(RELATIVE_SITE_PACKAGES "Lib/site-packages")
else()
    # On Linux, dynamically extract it by matching everything after the 'lib/' folder
    string(REGEX MATCH "lib/python[0-9]+\\.[0-9]+/site-packages" RELATIVE_SITE_PACKAGES "${ABS_PYTHON_SITELIB}")
endif()
message(STATUS "ABS_PYTHON_SITELIB: ${ABS_PYTHON_SITELIB}")
message(STATUS "RELATIVE_SITE_PACKAGES: ${RELATIVE_SITE_PACKAGES}")

# Ask Python where its Standard Library is
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('stdlib'))"
    OUTPUT_VARIABLE STDLIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

file(TO_CMAKE_PATH "${STDLIB_DIR}" STDLIB_DIR)

option(CPFD_ENABLE_PYTHON_WARNINGS "Enable and display Python runtime warnings" OFF)

if(CPFD_ENABLE_PYTHON_WARNINGS)
    message(STATUS "Python warnings are ENABLED (Development Mode)")
    add_compile_definitions(PYTHON_WARNINGS_ENABLED=1)
else()
    message(STATUS "Python warnings are STRIPPED/SILENCED (Production Mode)")
    add_compile_definitions(PYTHON_WARNINGS_ENABLED=0)
endif()

# --- 3. PATH SETUP ---
set(PACKAGE_PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/CPFD-ROM_Binary")
set(PY_ENTRY_POINT "${PACKAGE_PARENT_DIR}/cpfd_rom/rom_cli.py")
set(NUITKA_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/nuitka_out")
set(RELEASE_DIR "${CMAKE_CURRENT_BINARY_DIR}/Release")

if(WIN32)
    set(NUITKA_LIB "${NUITKA_OUTPUT_DIR}/rom_cli.lib")
else()
    set(NUITKA_LIB "${NUITKA_OUTPUT_DIR}/rom_cli.a")
endif()

set(NUITKA_FLAGS
    --module
    --include-package=cpfd_rom
    --follow-import-to=cpfd_rom
    --deployment
    --output-dir="${NUITKA_OUTPUT_DIR}"
    --report=${CMAKE_CURRENT_BINARY_DIR}/compilation_report.json
    --jobs=${N_CORES}
)
if(WIN32)
    list(APPEND NUITKA_FLAGS "--msvc=latest")
endif()

add_custom_command(
    OUTPUT "${NUITKA_LIB}"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${PACKAGE_PARENT_DIR}"
            "${Python3_EXECUTABLE}" -m nuitka
            ${NUITKA_FLAGS}
            "${PY_ENTRY_POINT}"
    DEPENDS "${PY_ENTRY_POINT}" "${REQUIREMENTS_FILE}"
    WORKING_DIRECTORY "${PACKAGE_PARENT_DIR}"
    USES_TERMINAL
    COMMENT "Compiling Python logic into a Static Library for C++ embedding"
)

add_custom_target(nuitka_static_target DEPENDS "${NUITKA_LIB}")

add_executable(cpfd_rom_cli)

set_target_properties(cpfd_rom_cli PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

cpfd_configure_version_header(cpfd_rom_cli)

target_sources(cpfd_rom_cli
    PRIVATE
        exeinfo.cpp
        main.cpp
        pythonmanager.cpp
    PUBLIC
        exeinfo.h
        pythonmanager.h

    PRIVATE
        "${PACKAGE_PARENT_DIR}/cpfd_rom/rom_cli.py"
)

target_link_libraries(cpfd_rom_cli
    PRIVATE
    Python3::Python           # The Interpreter Core
    PUBLIC
    licensemanager
    ${Boost_LIBRARIES}
)

target_include_directories(cpfd_rom_cli PRIVATE
    ${Python3_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/src"     # Allows #include "libdir/header.h"
    "${CMAKE_BINARY_DIR}/include"        # Path for generated headers
    )
target_include_directories(cpfd_rom_cli SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})

# Ensure Nuitka finishes before the C++ linker starts
add_dependencies(cpfd_rom_cli nuitka_static_target)

if(UNIX)
    # Force CMake to compile the binary using RUNPATH (new-dtags)
    # so that downstream dlopen() calls inherit the search directories.
    target_link_options(cpfd_rom_cli PRIVATE "-Wl,--enable-new-dtags")
endif()

#rom tool itself
install(TARGETS cpfd_rom_cli COMPONENT cpfd_rom_tool DESTINATION "bin")
set_target_properties(cpfd_rom_cli PROPERTIES BUILD_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/torch/lib")
set_target_properties(cpfd_rom_cli PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib/torch/lib")

if(WIN32)
    set(NUITKA_LIB_EXT "pyd")
    set(PY_TAG "cp${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
    set(NUITKA_LIB "rom_cli.${PY_TAG}-win_amd64.${NUITKA_LIB_EXT}")
else()
    set(NUITKA_LIB_EXT "so")
    # Linux standard PEP 3149 naming: cpython-312, cpython-313, etc.
    set(PY_TAG "cpython-${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}")
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH_TAG)
    set(NUITKA_LIB "rom_cli.${PY_TAG}-${ARCH_TAG}-linux-gnu.${NUITKA_LIB_EXT}")
endif()

# nuitka module
install(FILES "${NUITKA_OUTPUT_DIR}/${NUITKA_LIB}"
        DESTINATION "lib"
        RENAME "rom_cli.${NUITKA_LIB_EXT}"
        COMPONENT cpfd_rom_tool)


set(GLOBAL_EXCLUDES
   # PATTERN "__pycache__" EXCLUDE
    PATTERN "*.h" EXCLUDE
    PATTERN "*.hpp" EXCLUDE
    PATTERN "*.c" EXCLUDE
    PATTERN "*.cpp" EXCLUDE
    PATTERN "*.cuh" EXCLUDE
    PATTERN "*.h.in" EXCLUDE
    PATTERN "tests" EXCLUDE
    PATTERN "test" EXCLUDE
    PATTERN "_examples" EXCLUDE
    PATTERN "examples" EXCLUDE
    PATTERN "benchmarks" EXCLUDE
    PATTERN "categories" EXCLUDE
    PATTERN "cupy" EXCLUDE
    PATTERN "dask" EXCLUDE
    PATTERN "include" EXCLUDE
    PATTERN "share" EXCLUDE
)

set(STDLIB_EXCLUDES
    PATTERN "tkinter" EXCLUDE     # Graphical User Interface layout tool (Heavy!)
    PATTERN "ensurepip" EXCLUDE    # Internal pip bootstrapper (Unused at runtime)
    PATTERN "pydoc_data" EXCLUDE   # In-terminal documentation data
    PATTERN "idlelib" EXCLUDE      # Python's built-in text editor
    PATTERN "turtle.py" EXCLUDE    # Educational drawing module
    PATTERN "test" EXCLUDE         # Python's internal regression test suites
    PATTERN "sqlite3" EXCLUDE
    PATTERN "xmlrpc" EXCLUDE
    PATTERN "lib2to3" EXCLUDE
    PATTERN "wsgiref" EXCLUDE
    PATTERN "macholib" EXCLUDE
    PATTERN "venv" EXCLUDE
)

#python dll / so
if(WIN32)
    get_target_property(PYTHON_DLL_PATH Python3::Python IMPORTED_LOCATION)
    get_filename_component(PYTHON_BIN_DIR "${PYTHON_DLL_PATH}" DIRECTORY)

    install(FILES
            "${PYTHON_DLL_PATH}"
            "${PYTHON_BIN_DIR}/python3.dll"
            DESTINATION "bin"
            COMPONENT cpfd_rom_tool)

else()
    get_target_property(PYTHON_SO_PATH Python3::Python IMPORTED_LOCATION)
    get_filename_component(REAL_PYTHON_SO_PATH "${PYTHON_SO_PATH}" REALPATH)

    message(STATUS "SUCCESS: Found Linux Python Shared Lib at: ${REAL_PYTHON_SO_PATH}")

    install(FILES "${REAL_PYTHON_SO_PATH}"
            DESTINATION "bin"
            COMPONENT cpfd_rom_tool)
endif()

#python stdlib
install(DIRECTORY "${STDLIB_DIR}/"
        DESTINATION "lib/python_stdlib"
        COMPONENT cpfd_rom_tool
        ${GLOBAL_EXCLUDES}
        ${STDLIB_EXCLUDES}
        )

if(WIN32)

# 1. Tell CMake to locate the required MSVC system runtime libraries
    include(InstallRequiredSystemLibraries)

    # 2. Stage them right at the root folder next to your main executable
    # This automatically populates the variable ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
    install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
            DESTINATION "bin"
            COMPONENT cpfd_rom_tool)

    install(DIRECTORY "${REAL_SYSTEM_DLLS}/"
            DESTINATION "lib/python_dlls"
            COMPONENT cpfd_rom_tool
            ${GLOBAL_EXCLUDES}
            )
else()
    # 1. Copy the system's actual lib-dynload binary extensions folder
    install(DIRECTORY "${REAL_SYSTEM_STDLIB}/lib-dynload"
            DESTINATION "lib/python_stdlib"
            COMPONENT cpfd_rom_tool
            ${GLOBAL_EXCLUDES}
            )

    # 2. Make sure you also grab the standard library pure .py files from the host system
    install(DIRECTORY "${REAL_SYSTEM_STDLIB}/"
            DESTINATION "lib/python_stdlib"
            COMPONENT cpfd_rom_tool
            FILES_MATCHING
            ${GLOBAL_EXCLUDES}
           ${STDLIB_EXCLUDES}
            PATTERN "*.py"
            )

    find_library(LIBFFI_PATH
        NAMES ffi libffi.so.6 libffi.so
        PATHS /usr/lib64 /usr/lib /usr/local/lib
    )

    if(LIBFFI_PATH)
        get_filename_component(REAL_LIBFFI_PATH "${LIBFFI_PATH}" REALPATH)
        message(STATUS "SUCCESS: Found libffi to bundle: ${REAL_LIBFFI_PATH}")
        install(FILES "${REAL_LIBFFI_PATH}"
                DESTINATION "lib"
                RENAME "libffi.so.6"
                COMPONENT cpfd_rom_tool)
    else()
        message(WARNING "Could not find libffi.so on the build host!")
    endif()

    find_library(LIBCRYPTO_PATH
        NAMES crypto libcrypto.so.1.1
        PATHS /usr/lib64 /usr/lib
    )

    find_library(LIBSSL_PATH
        NAMES ssl libssl.so.1.1
        PATHS /usr/lib64 /usr/lib
    )

    if(LIBCRYPTO_PATH AND LIBSSL_PATH)
        get_filename_component(REAL_LIBCRYPTO_PATH "${LIBCRYPTO_PATH}" REALPATH)
        get_filename_component(REAL_LIBSSL_PATH "${LIBSSL_PATH}" REALPATH)
        get_filename_component(REAL_LIBFFI_PATH "${LIBFFI_PATH}" REALPATH)

        message(STATUS "SUCCESS: Found OpenSSL 1.1 to bundle:")
        message(STATUS "  -> ${REAL_LIBCRYPTO_PATH}")
        message(STATUS "  -> ${REAL_LIBSSL_PATH}")

        install(FILES "${REAL_LIBCRYPTO_PATH}"
                DESTINATION "lib"
                RENAME "libcrypto.so.1.1"
                COMPONENT cpfd_rom_tool)

        install(FILES "${REAL_LIBSSL_PATH}"
                DESTINATION "lib"
                RENAME "libssl.so.1.1"
                COMPONENT cpfd_rom_tool)

        install(FILES "${REAL_LIBFFI_PATH}"
                DESTINATION "lib"
                RENAME "libffi.so.6"
                COMPONENT cpfd_rom_tool)

        target_link_libraries(cpfd_rom_cli PRIVATE
                    "${REAL_LIBCRYPTO_PATH}"
                    "${REAL_LIBSSL_PATH}"
                    "${REAL_LIBFFI_PATH}"
                )
    else()
        message(WARNING "Could not find OpenSSL 1.1 shared libraries on the build host!")
    endif()
endif()

# python packages
set(PKGS_TO_COPY yaml numpy pandas torch torch_geometric tqdm sklearn trimesh matplotlib typing_extensions torchgen pytz dateutil six sympy
                 pyarrow fastparquet certifi idna urllib3 requests markupsafe jinja2 psutil fsspec mpmath  xxhash  )

foreach(PKG ${PKGS_TO_COPY})
    # Define paths for both possibilities
    set(SRC_PKG_DIR "${VENV_DIR}/${RELATIVE_SITE_PACKAGES}/${PKG}")
    set(SRC_PKG_FILE "${VENV_DIR}/${RELATIVE_SITE_PACKAGES}/${PKG}.py")
    file(TO_CMAKE_PATH "${SRC_PKG_DIR}" CLEAN_PKG_DIR)
    file(TO_CMAKE_PATH "${SRC_PKG_FILE}" CLEAN_PKG_FILE)

    # Case A: It's a standard directory folder (e.g., numpy, torch)
    if(EXISTS "${CLEAN_PKG_DIR}" AND IS_DIRECTORY "${CLEAN_PKG_DIR}")
        message(STATUS "Packaging Python directory module: ${PKG}")
        install(DIRECTORY "${CLEAN_PKG_DIR}"
                DESTINATION "lib"
                COMPONENT cpfd_rom_tool
                ${GLOBAL_EXCLUDES}
                )

        # Check for companion binary libraries (.libs)
        set(SRC_LIBS_DIR "${VENV_DIR}/${RELATIVE_SITE_PACKAGES}/${PKG}.libs")
        file(TO_CMAKE_PATH "${SRC_LIBS_DIR}" CLEAN_LIBS_DIR)
        if(EXISTS "${CLEAN_LIBS_DIR}")
            install(DIRECTORY "${CLEAN_LIBS_DIR}"
                    DESTINATION "lib"
                    COMPONENT cpfd_rom_tool
                    ${GLOBAL_EXCLUDES}
                    )
        endif()

    # Case B: It's a pure-Python single file module (e.g., typing_extensions.py)
    elseif(EXISTS "${CLEAN_PKG_FILE}")
        message(STATUS "Packaging Python file module: ${PKG}.py")
        install(FILES "${CLEAN_PKG_FILE}"
                DESTINATION "lib"
                COMPONENT cpfd_rom_tool)

    else()
        message(WARNING "Package dependency asset not found: ${PKG}")
    endif()

    if(PKG STREQUAL "torch")
        message(STATUS "Packaging Heavyweight Binary Framework: Torch")
        install(DIRECTORY "${CLEAN_PKG_DIR}"
                DESTINATION "lib"
                COMPONENT cpfd_rom_tool
                FILES_MATCHING
                ${GLOBAL_EXCLUDES}
                PATTERN "*.so*" # Ensure all shared objects are aggressively retained
                PATTERN "*.py"
        )
    endif()
endforeach()

if (NOT WIN32)
    # 1. Standard install loop copies 'nvidia' normally
    install(DIRECTORY "${VENV_DIR}/${RELATIVE_SITE_PACKAGES}/nvidia"
            DESTINATION "lib"
            COMPONENT cpfd_rom_tool
            ${GLOBAL_EXCLUDES}
            )

    # 2. Add an installation rule to prune the dead weight right before packaging
    install(CODE [[
        message(STATUS "SLIMMING INSTALLER: Purging unused heavy CUDA binaries...")

        # Define the exact path inside your CPack staging area
        set(CUDA_LIB_DIR "${CMAKE_INSTALL_PREFIX}/lib/nvidia/cu13/lib")

        if(EXISTS "${CUDA_LIB_DIR}")
            # A. Clean out heavy static archive files (.a) which are never needed at runtime
            file(GLOB STATIC_ARCHIVES "${CUDA_LIB_DIR}/*.a")
            if(STATIC_ARCHIVES)
                file(REMOVE ${STATIC_ARCHIVES})
            endif()

            # B. Clean out unused optional shared libraries (.so)
            # Add or remove items from this list based on your testing!
            set(LIBS_TO_REMOVE
                "libcufftw"
                "libcusolver"
                "libcusolverMg"
                "libnvrtc-builtins"
                "libnvblas"
                "libcheckpoint"
                "libcufile_rdma"
                "libnvrtc-builtins.alt"
                "libnvrtc.alt"
                "libnvtx3interop"
                "libnvperf_host"
                "libpcsamplingutil"
                "libnvperf_target"
            )

            foreach(LIB_NAME ${LIBS_TO_REMOVE})
                file(GLOB TARGET_FILES "${CUDA_LIB_DIR}/${LIB_NAME}.so*")
                if(TARGET_FILES)
                    file(REMOVE ${TARGET_FILES})
                endif()
            endforeach()
        endif()
    ]] COMPONENT cpfd_rom_tool)

endif()
