1. **System Requirements**
   - Linux (primary supported platform)
   - Python 3.10+
   - CUDA 12.112.2
   - PyTorch 2.2.2 + cu121
   - NVIDIA RTX / A-series / V100 / GV100 or newer recommended

2. **Installation (Wheel-Based)**
   - Install using internal `.whl` file via `pip install`
   - Optional user-local installation
   - PATH setup if `rom-cli-bin` is not found

3. **End-to-End Workflow Overview**
   - Generate CFD training/test data (BVR)
   - Export ASCII data
   - Convert ASCII to binary
   - Train, infer, and evaluate ROMs

4. **Generate CFD Training and Test Data**
   - Use Barracuda Virtual Reactor v25.1.1+
   - Create multiple operating points (Rev1, Rev2, , Test1)
   - Each directory corresponds to one operating condition

5. **Export ASCII Data from Barracuda VR**
   - Use BVR GUI ? Post-Run ? Quick Macro Panel
   - Output all data to text
   - Preserve native file naming conventions

6. **Convert ASCII Data to Binary Format**
   - Use provided data conversion utilities
   - Configure `conversion_config.yaml`
   - Generate `_npy` or other binary directories

7. **Run the ROM CLI**
   - Execute ROM pipelines using `rom-cli-bin`
   - Driven entirely by YAML configuration

8. **YAML Configuration**
   - Specify ROM type, field type, variables
   - Define training directories and parameter mapping
   - Control training, inference, and graph rebuilding

9. **Output Directory Structure**
   - ROM results written under `rom_output/`
   - Separate directories for transient and time-averaged runs

10. **Post-Processing and Visualization**
    - Convert ROM output to Tecplot format
    - Load ROM and CFD data side-by-side for comparison

11. **ROM Accuracy Evaluation**
    - Compute correlation metrics
    - Evaluate transient and time-averaged fidelity
    - Generate plots and summary statistics

12. **Project Structure**
    - Modular layout for ML ROMs, PCA-RBF ROMs, utilities
    - CLI entry point and configuration templates

13. **Support**
    - Internal use only
    - Author and contact information
