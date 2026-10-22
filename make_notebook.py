"""Create the notebook entry point for training and inspecting the experiment."""

import nbformat as nbf

from data import ROOT


def main():
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# English-to-Darija translation with manual LSTM cells\n\nZouhair Eouichak, Aymane El Youssi, Malak Bartal and Sara Baalla. TP B.\n\nThe target uses the Latin-script `darija` column. Read README.md and REQUIREMENTS.md before running the experiment."
        ),
        nbf.v4.new_code_cell("%run test_models.py"),
        nbf.v4.new_markdown_cell(
            "## Train the nine configurations\nThis cell runs the complete benchmark and replaces the experiment files in results/. For a short pipeline check, use `--smoke --output-dir .build/smoke` instead."
        ),
        nbf.v4.new_code_cell("%run train.py --epochs 12"),
        nbf.v4.new_code_cell("%run verify_results.py\n%run make_figures.py"),
        nbf.v4.new_code_cell(
            "import pandas as pd\npd.read_csv('results/comparison.csv')"
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\ndisplay(Image('results/data_exploration.png'))\ndisplay(Image('results/learning_curves.png'))\ndisplay(Image('results/test_comparison.png'))"
        ),
    ]
    nbf.write(notebook, ROOT / "Translation_Project.ipynb")


if __name__ == "__main__":
    main()
