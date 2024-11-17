# DAI_ReinforcementLearning


This project implements a traffic light control simulation using SUMO (Simulation of Urban MObility) with a Reinforcment Learning approach in **M**ulti**A**gent **S**ystem.

## Prerequisites

- Python 3.x
- SUMO (Simulation of Urban MObility)

## Installation

1. Clone the repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Make sure SUMO is properly installed and configured in your system

## Usage

The simulation can be run using the following command structure:

```bash
python src/main.py --env <environment_name> [--nogui] [--steps <number_of_steps>]
```

### Command Line Arguments

- `--env` (required): Specifies the simulation environment
  - Example environments: "2x2", "3x3"
  - Must correspond to a configuration file in `includes/sumo/<env_name>/main.sumocfg`

- `--nogui` (optional): Launches SUMO without GUI
  - Default: True (runs in GUI mode)
  - No value needed, just add the flag to disable GUI

- `--steps` (optional): Number of simulation steps to run
  - Default: 5000
  - Must be a positive integer

### Examples

1. Basic run with 2x2 environment:
```bash
python src/main.py --env 2x2
```
2. Basic run with 3x3 environment:
```bash
python src/main.py --env 3x3
```

## Project Structure




## Troubleshooting

If you encounter the error "argument --nogui: expected one argument", make sure you're using the correct command syntax. The --nogui flag doesn't require a value, just include it to disable the GUI.

## Contributing

[Add your contribution guidelines here]

## License

[Add your license information here]