#include "exeinfo.h"
#include "version.h"

#include "boost/program_options.hpp"

#include <iostream>

void ExeInfo::printVersionInformation()
{
    std::cout << fullName << " version: " << VER_NUM << std::endl;
    std::cout << CPFD_PRODUCT_NAME << " release: " << RELEASE_NUM << std::endl;
    std::cout << "License version: " << LicenseManager::getLicenseVersion() << std::endl;
}

void ExeInfo::printLicenseInformation(const LicenseManager& lm)
{
    std::cout << lm.getLicenseManagerName() << std::endl;
}

void ExeInfo::printDevelopmentInformation()
{
    std::cout << "Build time: " << BUILD_DATE << std::endl;
    std::cout << "Commit hash: " << BUILD_HASH << std::endl;
}

bool ExeInfo::setOptions(int argc, char* argv[], const LicenseManager& lm)
{
    namespace po = boost::program_options;
    po::options_description generalOptions(std::string(fullName) + " options");
    generalOptions.add_options() //
        ("help,h", "Display help information") //
        ("config", po::value<std::string>()->implicit_value(""), "Configuration input file") //
        ("version,v", po::value<std::string>()->implicit_value(""), "Display version information") //
        ("licensing", "Display license information") //
        ("infer-only", "Run in inference only mode") //
        ;

    po::options_description devOptions(std::string(fullName) + " developement options");
    devOptions.add_options() //
        ("versiond,d", ("Display " + std::string(fullName) + " version information with development information").c_str()) //
        ;

    po::options_description allOptions("Allowed options");
    allOptions.add(generalOptions).add(devOptions);

    // process options
    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, allOptions), vm);

        if (vm.count("help"))
        {
            std::cout << generalOptions << std::endl;
            return false;
        }
        if (vm.count("version"))
        {
            std::string versionArg = vm["version"].as<std::string>();
            if (versionArg.empty())
            {
                printVersionInformation();
            }
            else
            {
                std::cout << [&versionArg]() -> const char*
                {
                    if (versionArg == "major") {return VER_MAJOR;}
                    else if (versionArg == "feature") {return VER_FEATURE;}
                    else if (versionArg == "minor") {return VER_MINOR;}
                    else if (versionArg == "number") {return VER_NUM;}
                    else if (versionArg == "hash") {return BUILD_HASH;}
                    else if (versionArg == "releasehash") {return RELEASE_HASH;}
                    else if (versionArg == "builddate") {return BUILD_DATE;}
                    else if (versionArg == "perpetual") {return LicenseManager::getLicenseVersion();}
                    else {return "Unrecognized version option.";}
                }() << std::endl;
            }
            return false;
        }
        if (vm.count("versiond"))
        {
            printVersionInformation();
            printDevelopmentInformation();
            return false;
        }
        if (vm.count("licensing"))
        {
            printLicenseInformation(lm);
            return false;
        }
        if (vm.count("config"))
        {
            std::string versionArg = vm["config"].as<std::string>();
            if (versionArg.empty()) {
              std::cout << "An input file must be specified with the --config option" << std::endl;
              return false;
            }
            //pass this on to python for processing
        }
        if (vm.count("infer-only"))
        {
            inferOnly = true;
            //pass this on to python for processing
        }
        po::notify(vm);
        return true; // all was ok.
    }
    catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }

    // if no options provided, print tool information and help:
    printVersionInformation();
    std::cout << std::endl;
    std::cout << generalOptions << std::endl;
    return false;
}
