# The Hybrid Approach
> Tool implementing a hybrid approach for efficient decision-tree inference on embedded systems.

> **Note:** This repository was created to support the review of *On the Design of Decision Tree Visiting Kernels at the Edge*, but it is openly available and usable by the community.

## Overview
The Hybrid Approach is a strategy presented in the paper *On the Design of Decision Tree Visiting Kernels at the Edge*, designed for modern edge platforms equipped with cache hierarchies and tightly coupled memories (TCM).It relies on partitioning the original decision-tree-based model into two distinct sub-models, each mapped to a different memory region and executed using the kernel best suited to the characteristics of that memory.

Specifically, the sub-model allocated in TCM is executed using a Single Instruction Multiple Trees (SIMT) kernel, which enables parallel evaluation of multiple trees and fully exploits the low-latency and deterministic access of TCM.  
Conversely, the sub-model placed in Static Random Access Memory (SRAM) is executed using the DT-Rec kernel, a highly optimized and cache-friendly recursive implementation tailored for edge and resource-constrained scenarios.

This repository provides a tool that, starting from a dataset, generates C header code implementing this hybrid execution model, enabling efficient deployment of decision-tree inference on embedded systems.

## Key Features
- Implemented in Python and provided with a graphical user interface (GUI) for ease of use.
- Generates portable C code that can be directly included in embedded projects.
- Uses a novel Single Instruction Multiple Trees (SIMT) approach that exploits ARM Helium vector extensions to parallelize decision-tree execution.
- Supports both Random Forest and XGBoost models, handling classification and regression tasks.
- Implements a hybrid execution strategy by partitioning the model across different memory regions (TCM and SRAM).
- Employs 5-fold cross-validation during model training to improve robustness and generalization.
- After training, models are exported in `joblib` format, enabling fast reuse and integration with separate workflows.

## Project Structure
The repository is organized as follows:
```text
├── Code_Generator/
│ ├── GUI/
│ │ └── gui.py # Main Python module providing the graphical user interface
│ ├── Kernels/
│ │ └── ... # Modules for kernel-specific C code generation
│ ├── Utils/
│   └── ... # Utilities for model training and preprocessing
│ 
├── Datasets/
│ └── ... # Example datasets for testing and experimentation
├── README.md
└── LICENSE
```

The `Code_Generator` directory contains the Python modules composing the core of the tool.  
The `GUI` subdirectory provides the main entry point for interacting with the tool through a graphical interface.  
The `Kernels` directory includes the modules responsible for generating C code for the supported execution kernels.  
The `Utils` directory contains utilities for model training, validation, and preprocessing.  

The `Datasets` directory provides example datasets that can be used to quickly test and evaluate the tool.

## Requirements

This section outlines the requirements needed to use the tool and to integrate the generated code.

### Tool Requirements

To run the code generation tool, the following requirements must be satisfied:

- **Python version:** Python 3.13.5  
- A standard Python environment capable of running scientific and machine learning libraries.
- A system supporting graphical user interfaces to run the provided GUI.

It is recommended to use a virtual environment to manage dependencies and ensure reproducibility.

The tool has been tested on macOS Sequoia 15.6 and Windows 10 Pro.

#### Python Standard Library
The following modules are part of the Python standard library and do not require additional installation:

- `os`
- `sys`
- `csv`
- `json`
- `re`
- `shutil`
- `threading`
- `traceback`
- `pathlib`
- `typing`
- `collections`
- `io`

#### External Python Dependencies
The following external libraries are required to run the tool:

- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `scipy`
- `joblib`

All required external dependencies can be installed using `pip`:

```bash
pip install numpy pandas scikit-learn xgboost scipy joblib
```

#### GUI Support
The graphical user interface is implemented using `Tkinter`, which is included by default in most Python distributions.
If `Tkinter` is missing, it may need to be installed separately depending on the operating system.

### Generated Code Requirements

The generated code is provided as a C header file and can be included in any embedded C project.

To correctly compile and execute the generated code, the following requirements must be satisfied:

- A target platform based on an **ARM processor**.
- Support for **ARM Helium vector extensions**, as the code relies on a SIMD-based execution model.
- Compilation using an **ARM bare-metal toolchain**, such as `arm-none-eabi-gcc`.
- A memory architecture featuring **Static RAM (SRAM)** and an **L1 data cache**.
- Availability of **Tightly Coupled Memories (TCM)** to enable low-latency and deterministic execution of the parallel kernel.

No operating system is required, and the generated code can be integrated into bare-metal projects.

## Usage

To start the tool, launch the graphical user interface by running the following command from the project root:

```bash
python3 -m Code_Generator.GUI.gui
```




