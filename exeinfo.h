#ifndef EXEINFO_H
#define EXEINFO_H

#include "license-manager/license-manager/licensemanager.h"

struct ExeInfo
{
    const char* fullName = "CPFD Rom Tool";

    void printVersionInformation();
    void printLicenseInformation(const LicenseManager &lm);
    void printDevelopmentInformation();
    bool setOptions(int argc, char *argv[], const LicenseManager &lm);
};

#endif // EXEINFO_H
