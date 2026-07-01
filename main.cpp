#include "pythonmanager.h"

#include "exeinfo.h"

#include <iostream>
#include "license-manager/license-manager/licensemanager.h"

// todo, require ROM license
License::Type licenseType = License::Type::CpfdRomTraining;

int main(int argc, char* argv[]) {

#ifdef __linux__
    // Prevent the bundled Rocky 8 libcrypto from locking up inside modern host environments
    setenv("OPENSSL_ENABLE_FIPS", "0", 1);
    setenv("OPENSSL_SYSTEM_CIPHERS_OVERRIDE", "xyz", 1);
    setenv("OPENSSL_CONF", "/etc/pki/tls/openssl.cnf", 1);
#endif

  LicenseManager licenseManager("romtool", std::cout, true);
  ExeInfo exeInfo;

  if (!exeInfo.setOptions(argc, argv, licenseManager)) exit(0);

  exeInfo.printVersionInformation();
  std::cout << std::endl;

  licenseManager.checkoutLicenses({ { licenseType, 1 } });
  std::cout << std::endl;

  if (licenseManager.countValidLicenses(licenseType) >= 1) {
    try {
      // Initialize and run Python
      PythonManager py(argc, argv);
      if (!py.runModule("rom_cli", "main")) {
        return 1;
      }
    }
    catch (const std::exception& e) {
      std::cerr << "Fatal Error: " << e.what() << std::endl;
      return 1;
    }
  } else {
      std::cerr << "Unable to checkout " << exeInfo.fullName << " license" << std::endl;
      licenseManager.displayHelpMessage();
      return 2;
  }

  return 0;
}
