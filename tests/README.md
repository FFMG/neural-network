# Neural Network Unit Tests

This directory contains unit tests for the Neural Network library using the **Google Test** framework. The project is configured using **CMake**, allowing it to be built across different platforms and IDEs.

## How to open and run with Visual Studio 2022

Visual Studio has native support for CMake projects. You do not need to install CMake separately.

1.  Open Visual Studio 2022.
2.  Go to **File > Open > Folder...**.
3.  Navigate to and select this `tests` folder.
4.  Visual Studio will automatically detect `CMakeLists.txt` and begin the "CMake Generation" phase. During this time, it will download Google Test from GitHub.
5.  Once the generation is complete, go to the **Build** menu and select **Build All**.
6.  To run the tests, open the **Test Explorer** (**Test > Test Explorer**) and click **Run All Tests**, or select `neuralnetwork_tests.exe` as your startup item and press **F5**.

## How to open and run with GCC/G++ (Command Line)

To build and run the tests using GCC/G++ on Linux or via MinGW/WSL on Windows, ensure you have `cmake` and `make` installed.

1.  Open a terminal in this `tests` directory.
2.  Create a build directory and navigate into it:
    ```bash
    mkdir build
    cd build
    ```
3.  Configure the project:
    ```bash
    cmake ..
    ```
    *Note: This will download Google Test if it's not already present.*
4.  Build the tests:
    ```bash
    make
    ```
5.  Run the test executable:
    ```bash
    ./neuralnetwork_tests
    ```
