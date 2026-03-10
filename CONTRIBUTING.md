# Contributing

Thank you for your interest in contributing! Here's how to get started.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/terratorch-building-segmentation.git
   cd terratorch-building-segmentation
   ```
3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies** (conda recommended for GDAL):
   ```bash
   conda create -n terratorch-seg python=3.10
   conda activate terratorch-seg
   conda install -c conda-forge gdal
   pip install -r requirements.txt
   pip install flake8 black pytest
   ```

## Development Workflow

1. Make your changes in the feature branch
2. Write or update tests in `tests/`
3. Run checks before committing:
   ```bash
   black src/ --line-length 120   # Format code
   flake8 src/                     # Lint
   pytest tests/ -v                # Run tests
   ```
4. Commit with DCO sign-off (required):
   ```bash
   git commit -s -m "feat: add new decoder configuration"
   ```
5. Push and open a Pull Request

## Commit Message Convention

| Prefix | Use |
|--------|-----|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or updating tests |
| `refactor:` | Code restructuring |
| `config:` | New experiment configuration |

## Adding New Experiment Configs

1. Create a new YAML file in `configs/`
2. Follow the structure of existing configs (trainer, model, data sections)
3. Document the experiment in the README results table
4. Run the config validation test: `pytest tests/test_config.py -v`

## Questions?

Open a [Discussion](https://github.com/OMUZ9924/terratorch-building-segmentation/discussions) or email [arbouzmaamar@gmail.com](mailto:arbouzmaamar@gmail.com).
